import pandas as pd

class ComponentDependencyModel:

    def __init__(self, path_to_transition_matrix: str = r'transition_matrix.csv'):
        self.transition_matrix = pd.read_csv(path_to_transition_matrix)


    def fix_fail_probability(self, failing_component_type: str, all_failing_component_types: list) -> float:
        '''
        Returns the probability for being able to fix a component that depends on other components.
        :param failing_component_type: The type of the component we're looking to fix
        :param all_failing_component_types: A list of component types which are currently failing.
        :return: The probability that the fix will fail.
        '''

        # reduce matrix to only the component and its failing components
        tm = self.transition_matrix  # for readability
        fail_probabilities = tm.loc[(tm[tm.columns[0]] == failing_component_type)][all_failing_component_types]
        return sum(fail_probabilities.values.tolist()[0])
