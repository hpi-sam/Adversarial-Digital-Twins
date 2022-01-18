import random
from typing import List
import pandas as pd
import numpy as np
from mrubis_controller.entities.observation import Issue
from mrubis_controller.entities.component_failure import ComponentFailure
from mrubis_controller.entities.components import Components

from mrubis_controller.entities.observation import Observation

class DigitalTwin:
    def __init__(self) -> None:
        self.build_propagation_matrix()
        self.build_utility_series()
        self.build_fix_cost_series()
        self.previous_observation = []
        self.issue_distribution = []

    def forward(self, last_observation, action):
        #TODO: Implement actual model
        # last_observation full state for the shop
        # predict next state
        # return next observation vector
        # return new full state for the shop
        pass

    def build_utility_series(self):
        self.utility_means = self.component_failure_series()
        self.utility_stds = self.component_failure_series()
    
    def build_fix_cost_series(self):
        self.fix_cost_means = self.component_failure_series()
        self.fix_cost_stds = self.component_failure_series()

    def component_failure_matrix(self) -> pd.DataFrame:
        failures = ComponentFailure.list()
        failures.remove(ComponentFailure.GOOD)
        components = Components.list()
        long_components = np.array([[component]*len(failures) for component in components]).flat
        num_components = len(long_components)
        arrays = [
            long_components,
            failures*len(components),
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["component", "status"])
        return pd.DataFrame(np.zeros((num_components, num_components)), index=index, columns=index)
    
    def component_failure_series(self) -> pd.Series:
        failures = ComponentFailure.list()
        failures.remove(ComponentFailure.GOOD)
        components = Components.list()
        long_components = np.array([[component]*len(failures) for component in components]).flat
        num_components = len(long_components)
        arrays = [
            long_components,
            failures*len(components),
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["component", "status"])
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
        # Count the number of other issues for each issue that is observed
        for observation in observations:
            for issue in observation.issues:
                utilities[issue.component_name][issue.failure_type].append(issue.utility)
                currentRow = self.propagation_matrix.loc[issue.component_name, issue.failure_type]
                for other_issue in observation.issues:
                    currentRow[other_issue.component_name][other_issue.failure_type] += 1
        # Compute the probabilites for each error
        self.compute_issue_injection_distribution()
        # Compute the probabilities of seeing another issue when we have one
        for index, row in enumerate(self.propagation_matrix.iterrows()):
            self.propagation_matrix.iloc[index] /= row[index]
            self.propagation_matrix.iloc[index][index] = 0
        # Compute utility means and standard deviations
        for idx in utilities.index:
            self.utility_means[idx] = np.mean(utilities[idx])
            self.utility_stds[idx] = np.std(utilities[idx])

    def get_next_issue(self) -> List[Issue]:
        failures = []
        # Get failed component
        failed_component, failure_type = np.random.choice(self.utility_means.index, self.utility_means.to_numpy())
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
            # TODO ADD FIXES AND COSTS
            issues.append(
                Issue(component_name=component, utility=utility, failure_type=failure, fixes=None)
            )
        # Get utilities
        
        


