import json
import logging
import random
import socket
from pathlib import Path
from json.decoder import JSONDecodeError
from subprocess import PIPE, Popen
from time import sleep

from component_utility_predictor import RidgeUtilityPredictor
from component_dependencies import ComponentDependencyModel

import pandas as pd
import numpy as np

logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)


class MRubisController():

    def __init__(self, host='localhost', port=8080, json_path='path.json') -> None:
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

        self.run_counter = 0
        self.number_of_shops = 0
        self.number_of_issues_per_shop = {}
        self.current_issue = {}
        self.mrubis_state = {}
        self.mrubis_state_history = []
        self.fix_history = []
        self.current_fixes = None
        self.socket = None
        self.mrubis_process = None
        self.utility_model = RidgeUtilityPredictor()
        self.output_path = Path(__file__).parent.resolve() / 'output'
        self.component_dependency_model = ComponentDependencyModel()

    def _start_mrubis(self):
        '''Launch mRUBiS as a subprocess. NOTE: Unstable. Manual startup from Eclipse is more robust.'''
        self.mrubis_process = Popen(
            self.launch_args,
            stdin=PIPE,
            stdout=PIPE,
            shell=False,
            cwd="../mRUBiS/ML_based_Control"
        )

    def _stop_mrubis(self):
        '''Terminate the mRUBiS process'''
        self.mrubis_process.terminate()

    def _connect_to_java(self):
        '''Connect to the socket opened on the java side'''
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sleep(1)
        self.socket.connect((self.host, self.port))
        logger.info('Connected to the Java side.')

    def _get_initial_state(self):
        '''Query mRUBiS for the number of shops, get their initial states'''
        self.number_of_shops = self._get_from_mrubis(
            'get_number_of_shops').get('number_of_shops')
        logger.info(f'Number of mRUBIS shops: {self.number_of_shops}')
        for _ in range(self.number_of_shops):
            shop_state = self._get_from_mrubis('get_initial_state')
            shop_name = next(iter(shop_state))
            self.mrubis_state[shop_name] = shop_state[shop_name]

    def _update_number_of_issues_in_run(self):
        '''Update the number of issues present in the current run'''
        self.number_of_issues_in_run = self._get_from_mrubis(
            'get_number_of_issues_in_run').get('number_of_issues_in_run')

    def _get_from_mrubis(self, message):
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

    @staticmethod
    def _get_info_from_issue(issue):
        '''Parse relevant information out of an issue dictionary'''
        shop_name = next(iter(issue))
        component_name = next(iter(issue[shop_name]))
        component_params = issue.get(shop_name).get(component_name)
        issue_name = component_params.get('failure_name')
        rules = [rule.strip()
                 for rule in component_params['rule_names'].strip('[]').split(',')]
        rule_costs = [
            float(costs) for costs in component_params['rule_costs'].strip('[]').split(',')]
        return shop_name, component_name, component_params, issue_name, rules, rule_costs

    def _pick_rule(self, issue, strategy='lowest'):
        '''Pick a rule to apply to an issue using a strategy'''
        _, component_name, _, _, rules, rule_costs = self._get_info_from_issue(
            issue)

        # ReplaceComponent can only be used on Authentication Service
        # TODO: Describe why can it only be used on Authentication Service?

        if component_name != 'Authentication Service':
            rule_costs = [cost for idx, cost in enumerate(
                rule_costs) if rules[idx] != 'ReplaceComponent']
            rules = [rule for rule in rules if rule != 'ReplaceComponent']

        if strategy == 'random':
            return random.choice(rules)
        if strategy == 'lowest':
            return rules[rule_costs.index(min(rule_costs))]
        if strategy == 'highest':
            return rules[rule_costs.index(max(rule_costs))]

    def _send_rule_to_execute(self, issue, rule):
        '''Send a rule to apply to an issue to mRUBiS'''
        shop_name, component_name, _, issue_name, _, _ = self._get_info_from_issue(
            issue)
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
        if self.components_fixed_in_this_run.get(shop_name) is None:
            self.components_fixed_in_this_run[shop_name] = []
        self.components_fixed_in_this_run[shop_name].append(
            (issue_name, component_name))

    def _predict_optimal_utility_of_fixed_components(self, fixes_can_fail):
        '''Predict the optimal utility of a component which should be fixed'''
        for shop, fixed_components in self.components_fixed_in_this_run.items():
            for issue_name, component in fixed_components:
                component_params = self.mrubis_state[shop][component]
                predicted_utility = self.utility_model.predict_on_mrubis_output(
                    pd.DataFrame(component_params, index=[0])
                )[0]
                if fixes_can_fail:
                    failing_component_types = [values[1] for component_info in self.components_fixed_in_this_run.values() for values in component_info]
                    type_of_component_to_fix = fixed_components[0][1]
                    fail_probability = self.component_dependency_model.fix_fail_probability(type_of_component_to_fix, failing_component_types)
                    predicted_utility = predicted_utility * (1-fail_probability)
                self.mrubis_state[shop][component]['predicted_optimal_utility'] = predicted_utility

    def __get_ranked_fix_instructions(self, state_df_before: pd.DataFrame, ranking_strategy):
        '''Get a list of tuples containing the fixes to apply ranked by a strategy'''
        rows_with_failure = state_df_before.dropna(
            subset=['predicted_optimal_utility'])
        if ranking_strategy == 'cost':
            rows_with_failure['min_cost'] = rows_with_failure['rule_costs'].apply(
                min)
            sorted_rows = rows_with_failure.sort_values(
                by='min_cost', ascending=True)
        elif ranking_strategy == 'utility':
            sorted_rows = rows_with_failure.sort_values(
                by='predicted_optimal_utility', ascending=False)
        elif ranking_strategy == 'random':
            sorted_rows = rows_with_failure.sample(frac=1)
        else:
            raise NotImplementedError(
                f'Strategy {ranking_strategy} is not implemented!')
        ranked_fix_instructions = sorted_rows.reset_index(level=0)\
                                             .set_index('failure_name', append=True)\
                                             .reorder_levels([0, 2, 1])\
                                             .index\
                                             .tolist()
        #We save the sorted fixes here, so we can use them for total utility calculations later.
        self.current_fixes = sorted_rows.reset_index()

        return ranked_fix_instructions

    def _get_component_fixing_order(self, state_df_before: pd.DataFrame, ranking_strategy: str = 'utility'):
        '''Get a complete list of all components to fix ranked by a ranking strategy'''
        ranked_fix_instructions_tuples = self.__get_ranked_fix_instructions(
            state_df_before, ranking_strategy)
        ranked_fix_instructions_dict = {
            shop: (issue, comp) for shop, issue, comp in ranked_fix_instructions_tuples}
        all_fix_order_tuples = []
        for shop, _ in ranked_fix_instructions_dict.items():
            for i_c_tuple in self.components_fixed_in_this_run[shop]:
                all_fix_order_tuples.append((shop, i_c_tuple[0], i_c_tuple[1]))
        return all_fix_order_tuples

    def _cumulative_utility_based_on_order(self, sorted_df: pd.DataFrame):
        n_rows = sorted_df.shape[0]
        cumulative_utility_per_component = [util * (n_rows - i) for i, util in enumerate(sorted_df['predicted_optimal_utility'])]
        sorted_df['cum_util'] = cumulative_utility_per_component
        return sorted_df

    def _send_order_in_which_to_apply_fixes(self, order_tuples):
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

    def _send_exit_message(self):
        '''Tell mRUBiS to stop its main loop'''
        self.socket.send("exit\n".encode("utf-8"))
        _ = self.socket.recv(64000)

    def _close_socket(self):
        '''Close the socket'''
        self.socket.close()

    def _state_to_df(self, fix_status):
        '''Return the current mRUBiS state as a pandas dataframe'''
        state_df = pd.DataFrame.from_dict({
            (fix_status, shop, component): self.mrubis_state[shop][component]
            for shop in self.mrubis_state.keys()
            for component in self.mrubis_state[shop].keys()},
            orient='index')
        self._set_shop_and_system_utility(state_df, fix_status)
        return state_df

    def _set_shop_and_system_utility(self, state_df, fix_status):
        '''Make the shop and system utility fields uniform across the run'''
        if fix_status == 'before':
            state_df['system_utility'] = state_df['system_utility'].min()
            state_df['shop_utility'] = state_df.groupby(
                level=[1])['shop_utility'].transform('min')
        if fix_status == 'after':
            state_df['system_utility'] = state_df['system_utility'].max()
            state_df['shop_utility'] = state_df.groupby(
                level=[1])['shop_utility'].transform('max')

    def _write_mrubis_state_history_to_disk(self, filename='mrubis'):
        '''Write the state history to disk'''
        history_df = pd.concat(self.mrubis_state_history, keys=np.repeat(
            np.arange(1, len(self.mrubis_state_history)+1), 2)).reset_index()
        history_df.columns = ['run', 'fix_status', 'shop',
                              'component'] + list(history_df.columns)[4:]
        self.output_path.mkdir(exist_ok=True)
        logger.info('Writing run history to disk...')
        history_df.to_csv(self.output_path / f'{filename}.csv', index=False)
        #history_df.to_excel(self.output_path / f'{filename}.xls', index=False)

    def _write_df_list_to_disk(self, df_list, filename='mHistory'):
        df = pd.concat(df_list).reset_index(drop=True)
        self.output_path.mkdir(exist_ok=True)
        df.to_csv(self.output_path / f'{filename}.csv', index=False)


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

    def _reset_predicted_utility(self):
        '''Reset the predicted utility field for components that have been fixed'''
        for shop, shop_components in self.mrubis_state.items():
            for component, _ in shop_components.items():
                self.mrubis_state[shop][component]['predicted_optimal_utility'] = np.nan

    def _remove_replaced_authentication_service(self):
        '''Remove instances of the Authentication Service component that have been replaced'''
        for shop, shop_components in self.mrubis_state.copy().items():
            if len([comp for comp in list(shop_components.keys()) if 'Authentication Service' in comp]) > 1:
                for component_type, component_params in shop_components.copy().items():
                    if 'Authentication Service' in component_type and np.isclose(float(component_params['component_utility']), 0):
                        del self.mrubis_state[shop][component_type]

    def run(self, external_start=False, max_runs=100, rule_picking_strategy='lowest', issue_ranking_strategy='utility', fixes_can_fail=False):
        '''Run and control mRUBiS for a maximum number of runs'''

        if not external_start:
            self._start_mrubis()
            if self.mrubis_process.poll() is None:
                logger.info('MRUBIS is running')

        # Account for Java being slow to start on some systems
        sleep(0.5)

        self._connect_to_java()

        # Train the model on the provided batch file
        self.utility_model.load_train_data()
        self.utility_model.train_on_batch_file()

        while self.run_counter < max_runs:
            self.run_counter += 1
            self.components_fixed_in_this_run = {}
            self.number_of_issues_handled_in_this_run = 0
            self.number_of_issues_in_run = 1  # make sure that the loop runs at least once
            #current_issues = []

            logger.info(f"Getting state {self.run_counter}/{max_runs}...")

            if self.run_counter == 1:
                self._get_initial_state()

            while self.number_of_issues_handled_in_this_run < self.number_of_issues_in_run:

                # Check how many issues there are (equals # of queries to send)
                self._update_number_of_issues_in_run()

                # Get the current issue to handle
                current_issue = self._get_from_mrubis('get_issue')
                #current_issues.append(current_issue)
                self._update_current_state(current_issue)

                # Pick rule, send it to mRUBiS
                picked_rule = self._pick_rule(
                    current_issue, strategy=rule_picking_strategy)
                self._send_rule_to_execute(current_issue, picked_rule)

                self.number_of_issues_handled_in_this_run += 1

            logger.info(
                f'Total number of issues to handle in current run: {self.number_of_issues_in_run}')
            logger.info(
                f'Total number of issues handled: {self.number_of_issues_handled_in_this_run}')

            # Predict the optimal utility of the components to fix...
            self._predict_optimal_utility_of_fixed_components(fixes_can_fail) #Side effect: [shop][component]['predicted_optimal_utility']
            state_df_before = self._state_to_df(fix_status='before')
            self.mrubis_state_history.append(state_df_before)

            # Get and send the order of the fixes to mRUBiS...
            component_fixing_order = self._get_component_fixing_order(
                state_df_before, ranking_strategy=issue_ranking_strategy)

            fixed_issues = self._cumulative_utility_based_on_order(self.current_fixes)
            self.fix_history.append(fixed_issues)

            logger.info(f'Fixing in this order: {component_fixing_order}')
            self._send_order_in_which_to_apply_fixes(component_fixing_order)

            # Fixes are now being applied...

            # Query the state of the affected components once more
            logger.info(
                "Getting state of affected components after taking action...")
            state_after_action = self._get_from_mrubis(
                message=json.dumps(
                    {shop_name: [i_c_tuple[1] for i_c_tuple in i_c_tuples]
                        for shop_name, i_c_tuples in self.components_fixed_in_this_run.items()}
                )
            )
            self._update_current_state(state_after_action)
            self._remove_replaced_authentication_service()
            self._reset_predicted_utility()

            state_df_after = self._state_to_df(fix_status='after')
            self.mrubis_state_history.append(state_df_after)

        self._send_exit_message()
        self._close_socket()

        if not external_start:
            self._stop_mrubis()

        self._write_mrubis_state_history_to_disk(
            filename=f'{max_runs}_runs_{rule_picking_strategy}_{issue_ranking_strategy}{"_fixingcanfail" if fixes_can_fail else ""}')

        self._write_df_list_to_disk(self.fix_history,
            filename=f'cumfix{max_runs}_runs_{rule_picking_strategy}_{issue_ranking_strategy}{"_fixingcanfail" if fixes_can_fail else ""}')


if __name__ == "__main__":

    controller = MRubisController()
    controller.run(external_start=True, max_runs=100,
                   rule_picking_strategy='highest', issue_ranking_strategy='random', fixes_can_fail=True)
