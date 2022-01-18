from failure_propagator.failure_propagator import FailureProgagator
from entities.components import Components
from entities.component_failure import ComponentFailure
from entities.fixes import Fixes
import numpy as np
import logging
import torch
import json
logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)

class Trainer():
    def __init__(self, agent, digital_twin=0, host='localhost', port=8080, json_path='path.json'):
        # Get instances of agent and digital twin
        self.agent = agent
        #self.digital_twin = digital_twin
        self.environment = FailureProgagator(host=host, port=port, json_path=json_path)

        self.mrubis_state = {}
        self.mrubis_utilities = {}
        self.utility_loss = torch.nn.MSELoss()
        self.fix_loss = torch.nn.MSELoss()
        self.utility_optimizer = torch.optim.Adam(self.agent.fix_predictor.parameters(), lr=0.001)

    def train(self, max_runs=1):
        run_counter = 0
        self._get_initial_observation()
        for shop_name, state in self.mrubis_state.items():
            self.mrubis_utilities[shop_name] = list(state.values())[0]['shop_utility']

        while run_counter < max_runs:
            number_of_issues = 1
            num_issues_handled = 0
            predicted_utilities = []

            # shop_name -> (fix vector, predicted utility)
            predicted_fixes_w_gradients = {}
            right_components_picked = {}
            predicted_fixes = []

            # Utilities of failed components before fixing
            failed_utilities = {}

            while num_issues_handled < number_of_issues:
                num_issues_handled+=1

                # Update number of issues
                number_of_issues = self.environment.get_number_of_issues_in_run()

                # Get current observation
                current_observation = self.environment.get_current_issues()
                logging.info("Current Observation:")
                logging.info(current_observation)

                # Update failed utilities
                for shop_name, state in current_observation.items():
                    failed_utilities[shop_name] = list(state.values())[0]['shop_utility']

                # predict the fix
                self.utility_optimizer.zero_grad()
                predicted_fix_vector, predicted_utility = self.agent.predict_fix(self.observation_to_vector(current_observation))
                predicted_utilities.append(predicted_utility.item())
                logging.info("Predictions: ")
                logging.info(predicted_fix_vector)
                logging.info(predicted_utility)

                # convert the fix to a json
                shop_name, failure_name, predicted_component, predicted_rule = self.vector_to_fix(predicted_fix_vector, current_observation)
                predicted_fixes.append({
                    'shop': shop_name,
                    'issue': failure_name,
                    'component': predicted_component
                })
                predicted_fixes_w_gradients[shop_name] = (predicted_fix_vector, predicted_utility)

                logging.info(f"We predicted fix: {shop_name, failure_name, predicted_component, predicted_rule}")

                # send the fix to mRubis
                right_components_picked[shop_name] = self.environment.send_rule_to_execute(shop_name, failure_name, predicted_component, predicted_rule)

            # predict the ranking of fixes
            order_indices = self.agent.predict_ranking(predicted_utilities)
            logger.info("We predicted order: ")
            logger.info(order_indices)

            # send fixes to mRubis
            self.environment.send_order_in_which_to_apply_fixes(predicted_fixes, order_indices)
            
            # getting the new state of the fixed components
            state_after_action = self.environment.get_from_mrubis(message=json.dumps({predicted_fix['shop']: [predicted_fix['component']] for predicted_fix in predicted_fixes}))
            utility_differences = {}
            for shop_name, state in state_after_action.items():
                try:
                    utility_differences[shop_name] = float(list(state.values())[0]['shop_utility']) - float(failed_utilities[shop_name])
                except:
                    continue
            logger.info(failed_utilities)
            logger.info(utility_differences)
            self._update_current_state(state_after_action)
            #TODO: Get reward & train predictors
            #TODO: Think of digital twin training logic
            run_counter+=1
            # self.train_agent(predicted_fixes_w_gradients, utility_differences)




        # gets data and calls train_agent and train_digital_twin

        #Loop:
            # train twin
            # train agent / or use exploration agent / use batch (both on real environment)

            # swtich to twin
            #Loop:
                # periodically execute "validation step" (short switch to the real environment) to detect if we can stay on the twin
                # break if we overfit
        pass

    def train_agent(self, predicted_fixes_w_gradients, utility_differences, right_components_picked):
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
        utility_loss = torch.tensor(0.0)
        fix_loss = torch.tensor(0.0)
        for shop_name, utility_difference in utility_differences.items():
            predicted_utility = predicted_fixes_w_gradients[shop_name][1]
            utility_loss += self.utility_loss(predicted_utility, torch.tensor(utility_difference))

            predicted_fix = predicted_fixes_w_gradients[shop_name][0]
            acceptance_threshold = 100
            max_index = torch.argmax(predicted_fix)
            idx_1, idx_2 = max_index.item() // predicted_fix.shape[1], max_index.item() % predicted_fix.shape[1]
            label = torch.zeros(predicted_fix.shape)

            if right_components_picked[shop_name] and utility_difference > acceptance_threshold:
                label[idx_1,idx_2] = 1.0
            elif right_components_picked[shop_name]:
                label[:,idx_2] = 1.0
                label[idx_1,idx_2] = 0.0


            fix_loss += self.fix_loss(predicted_fix, label)

        loss = utility_loss + fix_loss
        logging.info(f"Current loss: {loss.item()}, Utility loss: {utility_loss.item()}, Fix loss: {fix_loss.item()}")
        loss.backward()
        self.utility_optimizer.step()

            

            

    def _get_initial_observation(self):
        '''Query mRUBiS for the number of shops, get their initial states'''
        self.number_of_shops = self.environment.number_of_shops
        logger.info(f'Number of mRUBIS shops: {self.number_of_shops}')
        for _ in range(self.number_of_shops):
            shop_state = self.environment.get_initial_state()
            shop_name = next(iter(shop_state))
            self.mrubis_state[shop_name] = shop_state[shop_name]

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

    def train_digital_twin(self, data):
        pass

    def observation_to_vector(self, observation):
        all_components_list = Components.list()
        all_failures_list = ComponentFailure.list()
        failed_components = list(list(observation.values())[0].keys())
        failure_names = [dictionary['failure_name'] for dictionary in list(list(observation.values())[0].values())]
        failed_vector = np.zeros((len(all_failures_list), len(all_components_list)))
        for index, component in enumerate(all_components_list):
            if component in failed_components:
                failed_vector[all_failures_list.index(failure_names[failed_components.index(component)]), index] = 1
            else:
                failed_vector[all_failures_list.index(ComponentFailure.GOOD.value), index] = 1
        return failed_vector

    def vector_to_fix(self, fix_vector, observation):
        all_components_list = Components.list()
        all_fixes_list = Fixes.list()
        max_index = torch.argmax(fix_vector)
        idx_1, idx_2 = max_index.item() // fix_vector.shape[1], max_index.item() % fix_vector.shape[1]
        predicted_component = all_components_list[int(idx_2)]
        predicted_rule = all_fixes_list[int(idx_1)]
        shop_name = list(observation.keys())[0]
        failure_name = list(list(observation.values())[0].values())[0]['failure_name']

        return shop_name, failure_name, predicted_component, predicted_rule