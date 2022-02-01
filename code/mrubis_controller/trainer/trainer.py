from copy import deepcopy
import sys
from time import time
from typing import Dict, List, Union

import re
from agent.agent import Agent
from digital_twin.digital_twin import DigitalTwin
from entities.observation import AgentFix, ShopIssue, Observation, Issue, Fix, AppliedFix, InitialState, Component
from failure_propagator.failure_propagator import FailureProgagator
from entities.components import Components
from entities.component_failure import ComponentFailure
from entities.fixes import Fixes
import numpy as np
import logging
import torch
import os
import json
import wandb

logging.basicConfig(filename='training.log', filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

list_string_regex = re.compile(r"[\[(?:\,\s)]?([A-z]+)[\]\,]")
list_float_regex  = re.compile(r"[\[(?:\,\s)]?([\d\.]+)[\]\,]")

def map_observation(mrubis_observation: Dict) -> Observation:
    broken_components = list(mrubis_observation.values())
    issues = []
    for name, details in [next(iter(x.items())) for x in broken_components]:
        issue = Issue(
            component_name=name,
            utility=float(details["component_utility"]),
            failure_type=details["failure_name"],
            fixes=[Fix(fix_type=rule_name, fix_cost=float(rule_cost)) for rule_name, rule_cost in zip(list_string_regex.findall(details["rule_names"]), list_float_regex.findall(details["rule_costs"]))],
            shop_utility=0.0,
            criticality=float(details["criticality"]),
            importance=float(details["importance"]),
            reliability=float(details["reliability"])
        )
        issues.append(issue)
    return Observation(
        shop_name=next(iter(mrubis_observation.keys())),
        shop_utility=float(list(broken_components[0].values())[0]['shop_utility']),
        issues=issues,
        applied_fix=None
    )

def map_initial_state(initial_state: Dict[str, Dict[str, Union[str, float]]]) -> Dict[str, InitialState]:
    inital_state_result = {}
    for shop_name, shop_details in initial_state.items():
        components = [Component(
            component_name=c_name,
            utility=float(c_details["component_utility"]),
            criticality=float(c_details["criticality"]),
            importance=float(c_details["importance"]),
            reliability=float(c_details["reliability"])
        ) for c_name, c_details in shop_details.items()]
        shop_state = InitialState(
            shop_name=shop_name,
            shop_utility=float(next(iter(shop_details.values()))["shop_utility"]),
            components=components
        )
        inital_state_result[shop_name] = shop_state
    return inital_state_result


class Trainer():
    def __init__(self, agent: Agent, digital_twin: DigitalTwin, host='localhost', port=8080, json_path='path.json'):
        # Get instances of agent and digital twin
        self.agent = agent
        self.digital_twin = digital_twin
        self.environment = FailureProgagator(host=host, port=port, json_path=json_path)
        self.train_real = True # Train witht the real environment
        self.mrubis_state = {}
        self.mrubis_utilities = {}
        self.utility_loss = torch.nn.MSELoss()
        self.fix_loss = torch.nn.MSELoss()
        self.utility_optimizer = torch.optim.RMSprop(self.agent.fix_predictor.parameters(), lr=0.00001) # torch.optim.Adam(self.agent.fix_predictor.parameters(), lr=0.002)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.utility_optimizer, step_size=300, gamma=0.1)
        

    def train(self, max_runs=500, num_exploration=50):
        #os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project="test-project", entity="adversial-digital-twins") 
        wandb.config.update({
            "learning_rate": 0.01,
            "epochs": max_runs,
            "batch_size": 1
        })
        run_counter = 0
        self._get_initial_observation()
        for shop_name, state in self.mrubis_state.items():
            self.mrubis_utilities[shop_name] = list(state.values())[0]['shop_utility']

        observation_batch: List[Observation] = []
        explore = True
        while run_counter < max_runs:
            if run_counter != 0 and run_counter % 50 == 0:
                self.train_real = not self.train_real
                logging.info(f"train real: {self.train_real}")
            if run_counter == num_exploration:
                explore = False

            if self.train_real:
                logging.info(f"RUN {run_counter}")
                number_of_issues = 1
                num_issues_handled = 0
                predicted_utilities = []

                # shop_name -> (fix vector, predicted utility)
                predicted_fixes_w_gradients = {}
                right_components_picked = {}
                predicted_fixes = []

                # Utilities of failed components before fixing
                failed_utilities = {}
                observations_of_run: Dict[str, Observation] = {}

                while num_issues_handled < number_of_issues:
                    num_issues_handled+=1

                    # Update number of issues
                    number_of_issues = self.environment.get_number_of_issues_in_run()

                    # Get current observation
                    current_observation = self.environment.get_current_issues()
                    mapped_observaition = map_observation(current_observation)
                    observations_of_run[mapped_observaition.shop_name] = mapped_observaition

                    logging.debug("Current Observation:")
                    logging.debug(current_observation)

                    # Update failed utilities
                    for shop_name, state in current_observation.items():
                        failed_utilities[shop_name] = list(state.values())[0]['shop_utility']

                    # predict the fix
                    self.utility_optimizer.zero_grad()
                    predicted_fix_vector, predicted_utility = self.agent.predict_fix(self.observation_to_vector(current_observation), explore)
                    predicted_utilities.append(predicted_utility.item())
                    predicted_fixes_w_gradients[shop_name] = (predicted_fix_vector, predicted_utility)
                    logging.debug("Predictions: ")
                    logging.debug(predicted_fix_vector)
                    logging.debug(predicted_utility)

                    # convert the fix to a json
                    top_counter = 0
                    while True:
                        shop_name, failure_name, predicted_component, predicted_rule = self.vector_to_fix(predicted_fix_vector, current_observation, top_counter)

                        logging.debug(f"We predicted fix: {shop_name, failure_name, predicted_component, predicted_rule}")

                        # send the fix to mRubis
                        right_component_picked = self.environment.send_rule_to_execute(shop_name, failure_name, predicted_component, predicted_rule)
                        top_counter+=1
                        if right_component_picked:
                            
                            right_components_picked[shop_name] = {'right_component_predicted': top_counter==1, 'attempts': top_counter, 'right_component': predicted_component}
                            #print(right_components_picked[shop_name])
                            predicted_fixes.append({
                                'shop': shop_name,
                                'issue': failure_name,
                                'component': predicted_component
                            })
                            observations_of_run[shop_name].applied_fix = AppliedFix(fix_type=predicted_rule, fixed_component=predicted_component, worked=False, utility_after=0)
                            break


                # predict the ranking of fixes
                order_indices = self.agent.predict_ranking(predicted_utilities)

                # send fixes to mRubis
                self.environment.send_order_in_which_to_apply_fixes(predicted_fixes, order_indices)

                # getting the new state of the fixed components
                state_after_action = self.environment.get_from_mrubis(message=json.dumps({predicted_fix['shop']: [predicted_fix['component']] for predicted_fix in predicted_fixes}))
                utility_differences = {}
                if False:
                    wandb.log({"system utility after fix": float(list(state_after_action[predicted_fixes[0]['shop']].values())[0]['system_utility'])}, commit=False)
                    wandb.log({"sum shop utility after fix": float(sum([list(state_after_action[shop].values())[0]['shop_utility'] for shop in state_after_action.keys()]))}, commit=False)
                    wandb.log({"mean shop utility after fix": float(np.mean([list(state_after_action[shop].values())[0]['shop_utility'] for shop in state_after_action.keys()]))}, commit=False)
                for shop_name, state in state_after_action.items():
                    try:
                        wandb.log({"system utility after fix": float(list(state.values())[0]['system_utility'])}, commit=False)
                        utility_differences[shop_name] = float(list(state.values())[0]['shop_utility']) - float(failed_utilities[shop_name])
                        observations_of_run[shop_name].applied_fix.utility_after = float(list(state.values())[0]['shop_utility'])
                        if utility_differences > 0:
                            # We have a utility increase so the fix worked :)
                            observations_of_run[shop_name].applied_fix.worked = True
                        else:
                            # We don't have a utility increase so the fix did not work :(
                            observations_of_run[shop_name].applied_fix.worked = False
                    except:
                        continue
                logging.debug(failed_utilities)
                logging.debug(utility_differences)
                self._update_current_state(state_after_action)
                #TODO: Get reward & train predictors
                #TODO: Think of digital twin training logic
                run_counter+=1
                if not explore:
                    self.train_agent(predicted_fixes_w_gradients, utility_differences, right_components_picked)
                observation_batch += list(observations_of_run.values())
        
        # self.train_digital_twin(observation_batch)

        # gets data and calls train_agent and train_digital_twin
            else:
                if len(observation_batch) != 0:
                    self.train_digital_twin(observation_batch)
                    observation_batch = []
                logging.info(f"RUN {run_counter}")
                logging.info("Using DigitalTwin")
                predicted_utilities = []

                # shop_name -> (fix vector, predicted utility)
                predicted_fixes_w_gradients = {}
                right_components_picked = {}
                predicted_fixes = []

                # Utilities of failed components before fixing
                failed_utilities = {}
                observations_of_run: Dict[str, Observation] = {}

                for issue_num in range(self.digital_twin.get_number_of_issues_in_run()):

                    # Get current observation
                    current_observation = self.digital_twin.get_current_issues()

                    logging.debug("Current Observation:")
                    logging.debug(current_observation)

                    # Update failed utilities
                    for observation in current_observation:
                        failed_utilities[observation.shop] = observation.shop_utility

                    # predict the fix
                    self.utility_optimizer.zero_grad()
                    observation_vector = self.digital_twin_observation_to_vector2(current_observation)
                    predicted_fix_vector, predicted_utility = self.agent.predict_fix(observation_vector, False)
                    predicted_utilities.append(predicted_utility.item())
                    predicted_fixes_w_gradients[current_observation[0].shop] = (predicted_fix_vector, predicted_utility)
                    logging.debug("Predictions: ")
                    logging.debug(predicted_fix_vector)
                    logging.debug(predicted_utility)

                    # convert the fix to a json
                    top_counter = 0
                    while True:
                        shop_name, failure_name, predicted_component, predicted_rule, agent_fix = self.vector_to_digital_twin_fix(predicted_fix_vector, current_observation[0], top_counter)

                        logging.debug(f"We predicted fix: {shop_name, failure_name, predicted_component, predicted_rule}")

                        # send the fix to mRubis
                        right_component_picked = self.digital_twin.send_rule_to_execute(shop_name, failure_name, predicted_component, predicted_rule)
                        top_counter+=1
                        if right_component_picked:
                            
                            right_components_picked[shop_name] = {'right_component_predicted': top_counter==1, 'attempts': top_counter, 'right_component': predicted_component}
                            #print(right_components_picked[shop_name])
                            predicted_fixes.append({
                                'shop': shop_name,
                                'issue': failure_name,
                                'component': predicted_component
                            })
                            break
                order_indices = self.agent.predict_ranking(predicted_utilities)
                # send fixes to mRubis
                self.digital_twin.send_order_in_which_to_apply_fixes(predicted_fixes, order_indices)
                # getting the new state of the fixed components
                state_after_action = self.digital_twin.get_from_mrubis(message=json.dumps({predicted_fix['shop']: [predicted_fix['component']] for predicted_fix in predicted_fixes}))
                utility_differences = {}
                for shop_name, state in state_after_action.items():
                    try:
                        utility_differences[shop_name] = float(list(state.values())[0]) - float(failed_utilities[shop_name])
                    except:
                        continue
                logging.debug(failed_utilities)
                logging.debug(utility_differences)
                # self._update_current_state(state_after_action)
                #TODO: Get reward & train predictors
                #TODO: Think of digital twin training logic
                run_counter+=1
                self.train_agent(predicted_fixes_w_gradients, utility_differences, right_components_picked, True)

    def train_agent(self, predicted_fixes_w_gradients, utility_differences, right_components_picked, digitial_twin = False):
        # -------
        #   Optimzation step:
        # LOOP:
            # Predict fix & utility for shop
            # Send fix for each shop to environment (feedback: did we hit the right component?)
        # pass utilities to ranking model
        # predict ranking
        # send ranking to mRubis
        # calculate loss and optimize models with reward
        # -------
        utility_loss = 0
        fix_loss = 0
        avg_attempts = 0
        counter = 0
        utility_gain = 0
        for shop_name, utility_difference in utility_differences.items():
            counter+=1

            predicted_utility = predicted_fixes_w_gradients[shop_name][1]

            utility_scaling_factor = 10**(-5)
            new_utility_loss = self.utility_loss(predicted_utility, torch.tensor([utility_difference*utility_scaling_factor])) * (0.9**right_components_picked[shop_name]['attempts'])
            utility_loss += new_utility_loss
            utility_gain+=utility_difference

            predicted_fix = predicted_fixes_w_gradients[shop_name][0]
            acceptance_threshold = 1000
            avg_attempts += right_components_picked[shop_name]['attempts']
            correct_attempt_index = torch.topk(predicted_fix.view(-1), right_components_picked[shop_name]['attempts'])[1][-1].item()
            idx_1, idx_2 = correct_attempt_index // predicted_fix.shape[1], correct_attempt_index % predicted_fix.shape[1]
            label = torch.zeros(predicted_fix.shape)

            if utility_difference > acceptance_threshold:
                #label[:,idx_2] = 1.0
                label[idx_1,idx_2] = 10.0
            else:
                label[:,idx_2] = 1.0
                label[idx_1,idx_2] = 0.0

            new_fix_loss = self.fix_loss(predicted_fix, label)
            fix_loss += new_fix_loss

        avg_utility_gain = utility_gain/counter
        loss = utility_loss + 2*fix_loss + (( -(avg_utility_gain))/100000)*counter# + avg_attempts/20
        print(f"Current loss: {loss/counter}, Utility loss: {utility_loss/counter}, Fix loss: {fix_loss/counter}, Average needed attempts: {avg_attempts/counter}, Average Utility Gain: {utility_gain/counter}")
        #logging.info(f"Current loss: {loss}, Utility loss: {utility_loss}, Fix loss: {fix_loss}")
        wandb.log({"loss": loss/counter}, commit=False)
        wandb.log({"utility loss": utility_loss/counter}, commit=False)
        wandb.log({"fix loss": fix_loss/counter}, commit=False)
        wandb.log({"digital twin": int(digitial_twin)}, commit=False)
        wandb.log({"avg utility gain": avg_utility_gain}, commit=False)
        wandb.log({"average needed attempts": avg_attempts/counter})
        loss.backward()
        self.utility_optimizer.step()
        self.scheduler.step()

    def _get_initial_observation(self):
        '''Query mRUBiS for the number of shops, get their initial states'''
        self.number_of_shops = self.environment.number_of_shops
        logging.debug(f'Number of mRUBIS shops: {self.number_of_shops}')
        for _ in range(self.number_of_shops):
            shop_state = self.environment.get_initial_state()
            shop_name = next(iter(shop_state))
            self.mrubis_state[shop_name] = shop_state[shop_name]
        self.inital_state = deepcopy(self.mrubis_state)
        self.digital_twin.set_initial_state(map_initial_state(self.mrubis_state))

    def _update_current_state(self, incoming_state):
        '''Update the controller's current mRUBiS state with an incoming state'''
        '''TODO: This is not a state, but a data update. '''
        for shop, shop_components in incoming_state.items():
            for component_type, component_params in shop_components.items():
                if shop not in self.mrubis_state.keys():
                    self.mrubis_state[shop] = {}
                if component_type not in self.mrubis_state[shop].keys():
                    self.mrubis_state[shop][component_type] = {}
                for param, value in component_params.items():
                    self.mrubis_state[shop][component_type][param] = value

    def train_digital_twin(self, observations: List[Observation]) -> None:
        self.digital_twin.train(observations)

    def observation_to_vector(self, observation):
        all_components_list = Components.list()
        all_failures_list = ComponentFailure.list()
        failed_components = list(list(observation.values())[0].keys())
        failure_names = [dictionary['failure_name'] for dictionary in list(list(observation.values())[0].values())]

        criticalities = [float(dictionary['criticality']) for dictionary in list(list(observation.values())[0].values())]
        connectivities = [float(dictionary['connectivity']) for dictionary in list(list(observation.values())[0].values())]
        reliabilities = [float(dictionary['reliability']) for dictionary in list(list(observation.values())[0].values())]
        importances = [float(dictionary['importance']) for dictionary in list(list(observation.values())[0].values())]
        provided_interfaces = [float(dictionary['provided_interface']) for dictionary in list(list(observation.values())[0].values())]
        required_interfaces = [float(dictionary['required_interface']) for dictionary in list(list(observation.values())[0].values())]
        adts = [float(dictionary['adt']) for dictionary in list(list(observation.values())[0].values())]
        perf_maxes = [float(dictionary['perf_max']) for dictionary in list(list(observation.values())[0].values())]
        sat_points = [float(dictionary['sat_point']) for dictionary in list(list(observation.values())[0].values())]
        replicas = [float(dictionary['replica']) for dictionary in list(list(observation.values())[0].values())]
        requests = [float(dictionary['request']) for dictionary in list(list(observation.values())[0].values())]

        padded_criticalities =np.zeros(len(all_components_list))
        padded_connectivities =np.zeros(len(all_components_list))
        padded_reliabilities =np.zeros(len(all_components_list))
        padded_importances =np.zeros(len(all_components_list))
        padded_provided_interfaces =np.zeros(len(all_components_list))
        padded_required_interfaces =np.zeros(len(all_components_list))
        padded_adts =np.zeros(len(all_components_list))
        padded_perf_maxes =np.zeros(len(all_components_list))
        padded_sat_points =np.zeros(len(all_components_list))
        padded_replicas =np.zeros(len(all_components_list))
        padded_requests =np.zeros(len(all_components_list))

        failed_vector = np.zeros((len(all_failures_list)-1, len(all_components_list)))
        counter = 0
        for index, component in enumerate(all_components_list):
            
            if component in failed_components:     
                padded_criticalities[index] = criticalities[counter]
                padded_connectivities[index] = connectivities[counter]
                padded_reliabilities[index] = reliabilities[counter]
                padded_importances[index] = importances[counter]
                padded_provided_interfaces[index] = provided_interfaces[counter]
                padded_required_interfaces[index] = required_interfaces[counter]
                padded_adts[index] = adts[counter]
                padded_perf_maxes[index] = perf_maxes[counter]
                padded_sat_points[index] = sat_points[counter]
                padded_replicas[index] = replicas[counter]
                padded_requests[index] = requests[counter]
                counter+=1
                failed_vector[all_failures_list.index(failure_names[failed_components.index(component)])-1, index] = 1
            else:
                continue
                failed_vector[all_failures_list.index(ComponentFailure.GOOD.value), index] = 1

        numberic_vector = np.array([padded_criticalities, padded_connectivities, padded_reliabilities, padded_importances, padded_provided_interfaces, padded_required_interfaces, padded_adts, padded_perf_maxes, padded_sat_points, padded_replicas, padded_requests])
        return np.concatenate([failed_vector.reshape(-1), numberic_vector.reshape(-1)])
    
    # ToDo ändere dieses epische naming
    def digital_twin_observation_to_vector2(self, observation: List[ShopIssue]):
        all_components_list = Components.list()
        all_failures_list = ComponentFailure.list()
        failed_components = [obs.component_name for obs in observation]
        failure_names = [obs.failure_type for obs in observation]

        criticalities = [issue.criticality for issue in observation]
        connectivities = [self.inital_state[issue.shop][issue.component_name]["connectivity"] for issue in observation]
        reliabilities = [issue.reliability for issue in observation]
        importances = [issue.importance for issue in observation]
        provided_interfaces = [self.inital_state[issue.shop][issue.component_name]["provided_interface"] for issue in observation]
        required_interfaces = [self.inital_state[issue.shop][issue.component_name]["required_interface"] for issue in observation]
        adts = [self.inital_state[issue.shop][issue.component_name]["adt"] for issue in observation]
        perf_maxes = [self.inital_state[issue.shop][issue.component_name]["perf_max"] for issue in observation]
        sat_points = [self.inital_state[issue.shop][issue.component_name]["sat_point"] for issue in observation]
        replicas = [self.inital_state[issue.shop][issue.component_name]["replica"] for issue in observation]
        requests = [self.inital_state[issue.shop][issue.component_name]["request"] for issue in observation]

        padded_criticalities =np.zeros(len(all_components_list))
        padded_connectivities =np.zeros(len(all_components_list))
        padded_reliabilities =np.zeros(len(all_components_list))
        padded_importances =np.zeros(len(all_components_list))
        padded_provided_interfaces =np.zeros(len(all_components_list))
        padded_required_interfaces =np.zeros(len(all_components_list))
        padded_adts =np.zeros(len(all_components_list))
        padded_perf_maxes =np.zeros(len(all_components_list))
        padded_sat_points =np.zeros(len(all_components_list))
        padded_replicas =np.zeros(len(all_components_list))
        padded_requests =np.zeros(len(all_components_list))

        failed_vector = np.zeros((len(all_failures_list)-1, len(all_components_list)))
        counter = 0
        for index, component in enumerate(all_components_list):
            
            if component in failed_components:     
                padded_criticalities[index] = criticalities[counter]
                padded_connectivities[index] = connectivities[counter]
                padded_reliabilities[index] = reliabilities[counter]
                padded_importances[index] = importances[counter]
                padded_provided_interfaces[index] = provided_interfaces[counter]
                padded_required_interfaces[index] = required_interfaces[counter]
                padded_adts[index] = adts[counter]
                padded_perf_maxes[index] = perf_maxes[counter]
                padded_sat_points[index] = sat_points[counter]
                padded_replicas[index] = replicas[counter]
                padded_requests[index] = requests[counter]
                counter+=1
                failed_vector[all_failures_list.index(failure_names[failed_components.index(component)])-1, index] = 1
            else:
                continue
                failed_vector[all_failures_list.index(ComponentFailure.GOOD.value), index] = 1

        numberic_vector = np.array([padded_criticalities, padded_connectivities, padded_reliabilities, padded_importances, padded_provided_interfaces, padded_required_interfaces, padded_adts, padded_perf_maxes, padded_sat_points, padded_replicas, padded_requests])
        return np.concatenate([failed_vector.reshape(-1), numberic_vector.reshape(-1)])
    

    def digital_twin_observation_to_vector(self, observations: List[ShopIssue]):
        all_components_list = Components.list()
        all_failures_list = ComponentFailure.list()
        failed_components = [obs.component_name for obs in observations]
        failure_names = [obs.failure_type for obs in observations]
        failed_vector = np.zeros((len(all_failures_list), len(all_components_list)))
        for index, component in enumerate(all_components_list):
            if component in failed_components:
                failed_vector[all_failures_list.index(failure_names[failed_components.index(component)]), index] = 1
            else:
                failed_vector[all_failures_list.index(ComponentFailure.GOOD.value), index] = 1
        return failed_vector

    def vector_to_fix(self, fix_vector, observation, top_k):
        all_components_list = Components.list()
        all_fixes_list = Fixes.list()

        max_index = torch.topk(fix_vector.view(-1), top_k+1)[1][-1].item()

        idx_1, idx_2 = max_index // fix_vector.shape[1], max_index % fix_vector.shape[1]
        predicted_component = all_components_list[int(idx_2)]
        predicted_rule = all_fixes_list[int(idx_1)]
        shop_name = list(observation.keys())[0]
        failure_name = list(list(observation.values())[0].values())[0]['failure_name']
        agentfix = AgentFix(failure_name, predicted_rule)
        return shop_name, failure_name, predicted_component, predicted_rule

    def vector_to_digital_twin_fix(self, fix_vector, observation: Observation, top_k):
        all_components_list = Components.list()
        all_fixes_list = Fixes.list()

        max_index = torch.topk(fix_vector.view(-1), top_k+1)[1][-1].item()

        idx_1, idx_2 = max_index // fix_vector.shape[1], max_index % fix_vector.shape[1]
        predicted_component = all_components_list[int(idx_2)]
        predicted_rule = all_fixes_list[int(idx_1)]
        shop_name = observation.shop
        failure_name = observation.failure_type
        agentfix = AgentFix(failure_name, predicted_rule)
        return shop_name, failure_name, predicted_component, predicted_rule, agentfix