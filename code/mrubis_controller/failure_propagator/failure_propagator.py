import json
import logging
import socket
from pathlib import Path
from json.decoder import JSONDecodeError
from time import sleep
from typing import Union

from .messages import Messages

from .failure_propagation_HMM import FPHMM

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

    def _connect_to_java(self) -> socket.socket:
        '''Connect to the socket opened on the java side'''
        mrubis_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sleep(1)
        mrubis_socket.connect((self.host, self.port))
        logger.info('Connected to the Java side.')
        return socket

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
        return self.get_from_mrubis(Messages.GET_NUMBER_OF_SHOPS).get('number_of_shops')

    def get_initial_state(self):
        shop_state = self.get_from_mrubis(Messages.GET_INITIAL_STATE)
        #self.propagator.update_state(shop_state)
        return shop_state

    def get_number_of_issues_in_run(self):
        return self.get_from_mrubis(Messages.GET_NUMBER_OF_ISSUES_IN_RUN).get('number_of_issues_in_run')
    
    def get_current_issue(self):
        return self.get_from_mrubis(Messages.GET_CURRENT_ISSUE)

    def send_rule_to_execute(self, shop_name, issue_name, component_name, rule):
        '''Send a rule to apply to an issue to mRUBiS'''

        picked_rule_message = {shop_name: {issue_name: {component_name: rule}}}
        logger.info(
            f"{shop_name}: Handling {issue_name} on {component_name} with {rule}")
        logger.debug('Sending selected rule to mRUBIS...')
        self.socket.send(
            (json.dumps(picked_rule_message) + '\n').encode("utf-8"))
        logger.debug("Waiting for mRUBIS to answer with 'rule_received'...")
        data = self.socket.recv(64000)
        if data.decode('utf-8').strip() == 'rule_received':
            logger.debug('Rule transmitted successfully.')
        # Remember components that have been fixed in this run

    def send_order_in_which_to_apply_fixes(self, order_tuples):
        '''Send the order in which to apply the fixes to mRUBiS'''
        logger.debug('Sending order in which to apply fixes to mRUBIS...')
        order_dict = {idx: {
            'shop': fix_tuple[0],
            'issue': fix_tuple[1],
            'component': fix_tuple[2]
        } for idx, fix_tuple in enumerate(order_tuples)}
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