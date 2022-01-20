import json
import logging
import socket
from pathlib import Path
from json.decoder import JSONDecodeError
from time import sleep
from typing import Union
import copy
from entities.messages import Messages
from failure_propagator.failure_propagation_HMM import FPHMM

logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)

class FailureProgagator():
    def __init__(self, host: str = 'localhost', port: int = 8080, json_path: str = 'path.json') -> None:
        '''Create a new instance of the mRubisController class'''

        # Put your command line here (In Eclipse: Run -> Run Configurations... -> Show Command Line)
        with open(json_path, 'r') as f:
            variable_paths = json.load(f)

        self.host = host
        self.port = port
        self.analytics = {}
        self.counter = 0
        self.last_real_issue = {}

        self.launch_args = [
            variable_paths['java_path'],
            '-DFile.encoding=UTF-8',
            '-classpath',
            variable_paths['dependency_paths'],
            '-XX:+ShowCodeDetailsInExceptionMessages',
            'mRUBiS_Tasks.Task_1',
        ]

        self.socket = self._connect_to_java()
        self.propagator = FPHMM()
        self.current_issues = []
        self.number_of_shops = self.get_from_mrubis(Messages.GET_NUMBER_OF_SHOPS).get('number_of_shops')

    def _connect_to_java(self) -> socket.socket:
        '''Connect to the socket opened on the java side'''
        mrubis_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sleep(1)
        mrubis_socket.connect((self.host, self.port))
        logger.debug('Connected to the Java side.')
        return mrubis_socket

    def get_from_mrubis(self, message: Union[Messages, str]):
        if isinstance(message, Messages):
            message = message.value

        '''Send a message to mRUBiS and return the response as a dictionary'''
        self.socket.send(f"{message}\n".encode("utf-8"))
        logger.debug(f'Waiting for mRUBIS to answer to message {message}')
        data = self.socket.recv(64000)

        try:
            mrubis_state = json.loads(data.decode("utf-8"))
        except JSONDecodeError:
            logger.error("Could not decode JSON input, received this:")
            logger.error(data)
            mrubis_state = "not available"

        return mrubis_state

    def get_number_of_shops(self):
        return self.number_of_shops

    def get_initial_state(self):
        shop_state = self.get_from_mrubis(Messages.GET_INITIAL_STATE)
        self.propagator.add_initial_state(shop_state)
        return shop_state

    def get_number_of_issues_in_run(self):
        return self.get_from_mrubis(Messages.GET_NUMBER_OF_ISSUES_IN_RUN).get('number_of_issues_in_run')
    
    def get_current_issues(self):
        # issues = {}
        # for num_shops in range(self.number_of_shops):
        #     num_issues = self.get_number_of_issues_in_run()
        #     if num_issues == 0:
        #         break
        issue = self.get_from_mrubis(Messages.GET_CURRENT_ISSUE)
        self.last_real_issue = copy.deepcopy(issue)

        #     issues = {**self.propagator.create_observation(issue), **issues}
        # self.current_issues = issues
        return self.propagator.create_observation(issue)
        # #print(issue)
        # component_name = list(list(issue.values())[0].keys())[0]
        # failure_name = list(issue.values())[0][component_name]["failure_name"]
        # rule_cost = list(issue.values())[0][component_name]["rule_costs"]
        # rule_names = list(issue.values())[0][component_name]["rule_names"]
        # component_utility = list(issue.values())[0][component_name]["component_utility"]
        # criticality = list(issue.values())[0][component_name]["criticality"]
        # importance = list(issue.values())[0][component_name]["importance"]
        # reliability = list(issue.values())[0][component_name]["reliability"]


        # #rule_names = issue.values()[0][component_name]["rule_names"]
        # if component_name not in self.analytics:
        #     self.analytics[component_name] = {}
        # if failure_name not in self.analytics[component_name]:
        #     self.analytics[component_name][failure_name] = {}
        #     self.analytics[component_name][failure_name]["component_utility"] = []
        #     self.analytics[component_name][failure_name]["criticality"] = []
        #     self.analytics[component_name][failure_name]["importance"] = []
        #     self.analytics[component_name][failure_name]["reliability"] = []
        # self.analytics[component_name][failure_name]["component_utility"].append(float(component_utility))
        # self.analytics[component_name][failure_name]["criticality"].append(float(criticality))
        # self.analytics[component_name][failure_name]["importance"].append(float(importance))
        # self.analytics[component_name][failure_name]["reliability"].append(float(reliability))

        # for name, cost in zip(rule_names[1:-1].replace(" ", "").split(","), rule_cost[1:-1].replace(" ", "").split(",")):
        #     if name not in self.analytics[component_name][failure_name]:
        #         self.analytics[component_name][failure_name][name] = []
        #     self.analytics[component_name][failure_name][name].append(float(cost)) 

        # self.counter += 1
        # print(self.counter)
        # if self.counter > 3000:

        #     for component_name in self.analytics.keys():
        #         for failure_name in self.analytics[component_name].keys():
        #             for rule_name in self.analytics[component_name][failure_name].keys():
        #                 self.analytics[component_name][failure_name][rule_name] = [np.mean(self.analytics[component_name][failure_name][rule_name]), np.std(self.analytics[component_name][failure_name][rule_name])]
        #                 self.analytics[component_name][failure_name]["component_utility"] = [np.mean(self.analytics[component_name][failure_name]["component_utility"]), np.std(self.analytics[component_name][failure_name]["component_utility"])]
        #                 self.analytics[component_name][failure_name]["criticality"] = [np.mean(self.analytics[component_name][failure_name]["criticality"]), np.std(self.analytics[component_name][failure_name]["criticality"])]
        #                 self.analytics[component_name][failure_name]["importance"] = [np.mean(self.analytics[component_name][failure_name]["importance"]), np.std(self.analytics[component_name][failure_name]["importance"])]
        #                 self.analytics[component_name][failure_name]["reliability"] = [np.mean(self.analytics[component_name][failure_name]["reliability"]), np.std(self.analytics[component_name][failure_name]["reliability"])]

        #     with open("rule_costs.json", "w") as file:
        #         json.dump(self.analytics, file, indent=2)
        
        #     sys.exit()
        # return issue
        # with open("test_issue.json", "w") as file:
        #     json.dump(self.propagator.create_observation(issue), file, indent=2)
        # sys.exit()
        

    def send_rule_to_execute(self, shop_name, failure_name, predicted_component, predicted_rule):
        '''Send a rule to apply to an issue to mRUBiS'''
        failed_components = list(list(self.last_real_issue.values())[0].keys())
        # print("Real failure: ")
        # print(failed_components)
        # print("Predicted component: ")
        # print(predicted_component)
        # print(predicted_component not in failed_components)
        if predicted_component not in failed_components:
            return False
       
        picked_rule_message = {shop_name: {failure_name: {predicted_component: predicted_rule}}}

        logger.debug(
            f"Handling {picked_rule_message}")
        logger.debug('Sending selected rule to mRUBIS...')
        self.socket.send(
            (json.dumps(picked_rule_message) + '\n').encode("utf-8"))
        logger.debug("Waiting for mRUBIS to answer with 'rule_received'...")
        data = self.socket.recv(64000)
        if data.decode('utf-8').strip() == 'rule_received':
            logger.debug('Rule transmitted successfully.')

        return True

        # Remember components that have been fixed in this run
        # logger.debug('Sending order in which to apply fixes to mRUBIS...')
        # order_dict ={
        #     0: {
        #     'shop': shop_name,
        #     'issue': issue_name,
        #     'component': component_name
        #         }
        #     }
        # self.socket.send((json.dumps(order_dict) + '\n').encode("utf-8"))
        # logger.debug(
        #     "Waiting for mRUBIS to answer with 'fix_order_received'...")
        # data = self.socket.recv(64000)
        # if data.decode('utf-8').strip() == 'fix_order_received':
        #     logger.debug('Order transmitted successfully.')

        # previous_shop_utility = float(list(self.current_issues[shop_name].items())[0]["shop_utility"])
        # current_issues = self.get_current_issues()
        # reward = float(list(self.current_issues[shop_name].items())[0]["shop_utility"]) - previous_shop_utility
        # return current_issues, reward


    def send_order_in_which_to_apply_fixes(self, predicted_fixes, order_indices):
        '''Send the order in which to apply the fixes to mRUBiS'''
        logger.debug('Sending order in which to apply fixes to mRUBIS...')
        order_dict = {index: predicted_fixes[index] for index in order_indices}
        logger.debug(order_dict)
        '''
        for issueComponent in order_dict:
            self.socket.send(json.dumps(issueComponent))
            data = self.socket.recv(64000)
        '''
        self.socket.send((json.dumps(order_dict) + '\n').encode("utf-8"))
        logger.debug(
            "Waiting for mRUBIS to answer with 'fix_order_received'...")
        data = self.socket.recv(64000)
        if data.decode('utf-8').strip() == 'fix_order_received':
            logger.debug('Order transmitted successfully.')

    def send_exit_message(self):
        '''Tell mRUBiS to stop its main loop'''
        self.socket.send("exit\n".encode("utf-8"))
        _ = self.socket.recv(64000)

    def close_socket(self):
        '''Close the socket'''
        self.socket.close()