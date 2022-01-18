import random
from typing import List, Union
import pandas as pd
import numpy as np
from mrubis_controller.entities.observation import Fix
from mrubis_controller.entities.fixes import Fixes
from mrubis_controller.entities.observation import Issue
from mrubis_controller.entities.component_failure import ComponentFailure
from mrubis_controller.entities.components import Components

from mrubis_controller.entities.observation import Observation

class ShopDigitalTwin:
    def __init__(self) -> None:
        self.build_propagation_matrix()
        self.build_utility_series()
        self.build_fix_cost_matrix()
        self.previous_observation = []
        self.issue_distribution = []
        self.real_failed_component: Union[None, Issue] = None
        self._is_fixed = True

    def apply_fix(self, fix):
        # TODO check if fix fixed the component
        pass

    def check_real(self, fix):
        # TODO check if the fix might fix the real broken component
        # if not then do the same as the failure propagator
        pass

    def is_fixed(self) -> bool:
        return self.is_fixed

    def build_utility_series(self):
        self.utility_means = self.component_failure_series()
        self.utility_stds = self.component_failure_series()
    
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
        utilities = self.component_failure_series()
        fix_lists = self.build_fix_lists()
        # Count the number of other issues for each issue that is observed
        for observation in observations:
            for issue in observation.issues:
                utilities[issue.component_name][issue.failure_type].append(issue.utility)
                costs_series = fix_lists.loc[issue.component_name, issue.failure_type]
                for fix in issue.fixes:
                    costs_series[fix.fix_type].append(fix.fix_cost)
                currentRow = self.propagation_matrix.loc[issue.component_name, issue.failure_type]
                for other_issue in observation.issues:
                    currentRow[other_issue.component_name][other_issue.failure_type] += 1
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
        failures = []
        # Get failed component
        failed_component, failure_type = np.random.choice(self.utility_means.index, self.issue_distribution)
        failures.append((failed_component, failure_type))
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
        return issues

class DigitalTwin:
    def __init__(self, shop_names: List[str]) -> None:
        self.shop_simulations = {shop_name: ShopDigitalTwin() for shop_name in shop_names}
    
    def train(self, observations: List[Observation]) -> None:
        for shop_name, sim in self.shop_simulations.items():
            sim.train(list(filter(lambda x: x.shop_name == shop_name, observations)))

    # TODO simulate protocol
