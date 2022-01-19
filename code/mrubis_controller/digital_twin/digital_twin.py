import json
import random
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from mrubis_controller.entities.observation import Fix, AgentFix, Issue, Observation, InitialState
from mrubis_controller.entities.fixes import Fixes
from mrubis_controller.entities.component_failure import ComponentFailure
from mrubis_controller.entities.components import Components

class ShopDigitalTwin:
    def __init__(self) -> None:
        self.build_propagation_matrix()
        self.build_utility_series()
        self.build_fix_cost_matrix()
        self.build_healthy_component_utilities()
        self.previous_observation = []
        self.issue_distribution = []
        self.real_failed_component: Union[None, Issue] = None
        self.correct_fix: Union[None, Fixes] = None
        self._is_fixed = True
        self.current_issues: Union[None, List[Issue]] = None
        self._healthy_shop_utility = 0.0

    def set_initial_state(self, inital_state: InitialState) -> None:
        assert len(inital_state.components) == len(Components.list())
        self._healthy_shop_utility = inital_state.shop_utility
        
    def get_shop_utility(self) -> float:
        if self.is_fixed():
            return self._healthy_shop_utility()
        assert self.current_issues != None
        failed_components = [issue.component_name for issue in self.current_issues]
        computed_utility = 0.0
        for component in Components.list():
            if component in failed_components:
                computed_utility += self.current_issues[failed_components.index(component)].utility
            else:
                computed_utility += self.healthy_component_utilities[component]
        return computed_utility

    def get_component_utility(self, component: Components) -> float:
        failed_component_names = [issue.component_name for issue in self.current_issues] if self.current_issues != None else []
        if self.is_fixed() or not component in failed_component_names:
            return self.healthy_component_utilities[component]
        else:
            return self.current_issues[failed_component_names.index(component)].utility

    def apply_fix(self, fix: AgentFix) -> None:
        if fix.component == self.real_failed_component:
            if fix.fix == self.correct_fix:
                self.current_issues = None
                self.real_failed_component = None
                self.correct_fix = None
                self.is_fixed = True

    def build_fix_success_rate(self) -> None:
        components = Components.list()
        fixes = Fixes.list()
        failures = ComponentFailure.list()
        failures.remove(ComponentFailure.GOOD)
        long_components = np.array([[comp]*len(failures)*len(fixes) for comp in components]).flatten().tolist()
        long_failures = np.array([[fail]*len(fixes) for fail in failures]).flatten().tolist() * len(components) 
        long_fixes = fixes * len(failures) * len(components)
        columns = pd.MultiIndex.from_arrays((long_components, long_failures, long_fixes))
        index = ['worked', 'failed']
        self.fix_success = pd.DataFrame(np.zeros((2, len(long_components))), index=index, columns=columns)
        self.fix_probs = pd.Series(np.zeros(len(long_components)), index=index)

    def reset_fix_success_rate(self) -> None:
        self.fix_success -= self.fix_success
        self.fix_probs -= self.fix_probs

    def check_real(self, fix_component: Components) -> bool:
        # TODO check if the fix might fix the real broken component
        # if not then do the same as the failure propagator
        return fix_component == self.real_failed_component

    def is_fixed(self) -> bool:
        return self.is_fixed

    def build_utility_series(self):
        self.utility_means = self.component_failure_series()
        self.utility_stds = self.component_failure_series()
    
    def build_healthy_component_utilities(self):
        components = Components.list()
        self.healthy_component_utilities = pd.Series(np.zeros(len(components)), index=components)

    def build_fix_cost_matrix(self):
        num_fail_comp = self.number_failure_components()
        num_fixes = len(Fixes.list())
        self.fix_cost_means = pd.DataFrame(
            np.zeros(num_fail_comp, num_fixes),
            index=self.build_component_failure_multi_index(),
            columns=Fixes.list()
        )
        self.fix_cost_stds = pd.DataFrame(
            np.zeros(num_fail_comp, num_fixes),
            index=self.build_component_failure_multi_index(),
            columns=Fixes.list()
        )
    
    def build_fix_lists(self) -> pd.DataFrame:
        num_fail_comp = self.number_failure_components()
        num_fixes = len(Fixes.list())
        return pd.DataFrame(
            np.array([[] for _ in range(num_fail_comp * num_fixes)]).reshape(num_fail_comp, num_fixes),
            index=self.build_component_failure_multi_index(),
            columns=Fixes.list()
        )

    def build_component_failure_multi_index(self) -> pd.MultiIndex:
        failures = ComponentFailure.list()
        failures.remove(ComponentFailure.GOOD)
        components = Components.list()
        long_components = np.array([[component]*len(failures) for component in components]).flat
        arrays = [
            long_components,
            failures*len(components),
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["component", "status"])
        return index
    
    def number_failure_components(self):
        return len(Components.list()) * (len(ComponentFailure.list()) - 1)

    def component_failure_matrix(self) -> pd.DataFrame:
        num_components = self.number_failure_components()
        index = self.build_component_failure_multi_index()
        return pd.DataFrame(np.zeros((num_components, num_components)), index=index, columns=index)
    
    def component_failure_series(self) -> pd.Series:
        num_components = self.number_failure_components()
        index = self.build_component_failure_multi_index()
        return pd.Series([[] for _ in range(num_components)], index=index)

    def build_propagation_matrix(self) -> None:
        """Builds the the propagation matrix.

        The propagation matrix has for every row the propability that the component in the column is
        also effected
        """
        self.propagation_matrix = self.component_failure_matrix()


    def reset_propagation_matrix(self) -> None:
        """Resets the propagation matrix to zero.
        """
        self.propagation_matrix -= self.propagation_matrix

    def compute_issue_injection_distribution(self):
        """Computes the probability that a component fails with a specific error.
        """
        self.issue_distribution = np.diag(self.propagation_matrix)
        self.issue_distribution /= self.issue_distribution.sum()

    def train(self, observations: List[Observation]) -> None:
        """Train the DigitalTwin using a batch ob observations that are sampled from
        an environment.

        Args:
            observations ([type]): [description]
        """
        self.previous_observation += observations
        self.reset_propagation_matrix()
        self.reset_fix_success_rate()
        utilities = self.component_failure_series()
        fix_lists = self.build_fix_lists()
        # Count the number of other issues for each issue that is observed
        for observation in observations:
            for issue in observation.issues:
                utilities[issue.component_name][issue.failure_type].append(issue.utility)
                costs_series = fix_lists.loc[issue.component_name, issue.failure_type]
                if issue.component_name == observation.applied_fix.fixed_component:
                    if observation.applied_fix.worked:
                        self.fix_success[observation.applied_fix.fixed_component, issue.failure_type, observation.applied_fix.fix_type]['worked'] += 1
                    else:
                        self.fix_success[observation.applied_fix.fixed_component, issue.failure_type, observation.applied_fix.fix_type]['failed'] += 1
                for fix in issue.fixes:
                    costs_series[fix.fix_type].append(fix.fix_cost)
                currentRow = self.propagation_matrix.loc[issue.component_name, issue.failure_type]
                for other_issue in observation.issues:
                    currentRow[other_issue.component_name][other_issue.failure_type] += 1
        # Compute the probability that a fix works on a broken component
        worked_np = self.fix_success.loc['worked'].to_numpy()
        failed_np = self.fix_success.loc['failed'].to_numpy()
        self.fix_probs.update(worked_np / (worked_np + failed_np))
        # Compute the probabilites for each error
        self.compute_issue_injection_distribution()
        # Compute the probabilities of seeing another issue when we have one
        for index, row in enumerate(self.propagation_matrix.iterrows()):
            self.propagation_matrix.iloc[index] /= row[index]
            self.propagation_matrix.iloc[index][index] = 0
        # Compute utility and fix costs means and standard deviations
        for idx in utilities.index:
            self.utility_means[idx] = np.mean(utilities[idx])
            self.utility_stds[idx] = np.std(utilities[idx])
            self.fix_cost_means[idx] = np.mean(fix_lists[idx])
            self.fix_cost_stds[idx] = np.std(fix_lists[idx])

    def get_next_issue(self) -> List[Issue]:
        if not self.is_fixed():
            return self.current_issues
        failures = []
        # Get failed component
        failed_component, failure_type = np.random.choice(self.utility_means.index, self.issue_distribution)
        failures.append((failed_component, failure_type))
        self.correct_fix = np.random.choice(self.fix_probs[failed_component, failure_type].index, self.fix_probs[failed_component, failure_type].to_numpy())
        # Get propagation
        # TODO: We assume that only one failure type per component was ever seen
        # This assumption is equal to that a cf<n> failure can only trigger a cf<n> failure where n=n
        for idx in self.propagation_matrix.columns:
            if random.random() <= self.propagation_matrix.iloc[failed_component, failure_type][idx]:
                failures.append(idx)
        # Compute utility and build issue
        issues = []
        for component, failure in failures:
            utility = np.random.normal(self.utility_means[component, failure], self.utility_stds[component, failure])
            # Get fixes that can be applied (which have a mean of greater than zero)
            # TODO use np.where here instead of if and iteration
            fixes = []
            for fix_name in self.fix_cost_means.columns:
                if self.fix_cost_means.iloc[component, failure][fix_name] != 0:
                    fixes.append(Fix(
                        fix_type=fix_name,
                        fix_cost=np.random.normal(
                            self.fix_cost_means.iloc[component, failure][fix_name],
                            self.fix_cost_std.iloc[component, failure][fix_name]
                        )
                    ))
            issues.append(
                Issue(component_name=component, utility=utility, failure_type=failure, fixes=fixes)
            )
            # Store the issue that was selected to be the initial failed component
            if component == failed_component:
                self.real_failed_component = issues[-1]
        self._is_fixed = False
        self.current_issues = issues
        return issues

class DigitalTwin:
    def __init__(self, shop_names: List[str]) -> None:
        self.shop_simulations = {shop_name: ShopDigitalTwin() for shop_name in shop_names}
        self.current_shop_index = 0
    
    def train(self, observations: List[Observation]) -> None:
        for shop_name, sim in self.shop_simulations.items():
            sim.train(list(filter(lambda x: x.shop_name == shop_name, observations)))

    def get_number_of_issues_in_run(self):
        counter = self._number_of_issues()
        
        if counter == 0:
            for sim in self.shop_simulations.values():
                sim.get_next_issue()
                counter += 1
        return counter
    
    def _number_of_issues(self) -> int:
        counter = 0
        for sim in self.shop_simulations.values():
            if not sim.is_fixed():
                counter += 1
        return counter

    def get_current_issues(self):
        simulations = list(self.shop_simulations.values())
        while self._number_of_issues > 0:
            self.current_shop_index = (self.current_shop_index + 1) % len(self.shop_simulations)
            sim = simulations[self.current_shop_index]
            if not sim.is_fixed():
                return sim.get_next_issue()
        return []

    def send_rule_to_execute(self, shop_name: str, failure_name: ComponentFailure, predicted_component: Components, predicted_rule: Fixes) -> bool:
        sim = self.shop_simulations[shop_name]
        if not sim.check_real(predicted_component):
            return False
        sim.apply_fix()
        return True

    def send_order_in_which_to_apply_fixes(self, predicted_fixes, order_indices):
        # TODO
        pass

    def get_from_mrubis(self, message: str) -> Dict:
        """Get the state of the given shops and components.

        Args:
            message (str): JSON string that maps shops to a list of components from which we
                           want to retrieve the utility from.

        Returns:
            Dict: A dict which maps shop to the component state.
        """
        message: Dict = json.loads(message)
        result = {}
        for shop, components in message.items():
            sim = self.shop_simulations[shop]
            for component in components:
                result[shop] = sim.get_shop_utility()
        return result
