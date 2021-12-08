import pandas as pd
import numpy as np
from typing import Dict

from .component_failure import ComponentFailure
from .components import Components

class FPHMM():
    def __init__(self, config_path: str ="moin"):
        #self.transition_matrix = pd.read_csv(config_path)
        self.current_state = None

    def get_state(self):
        return self.current_state
        
    def update_state(self, state):
        print(state)
        pass

    def create_observation(matrix: pd.DataFrame, failed_components: Dict[str, str]):
        # assumption: observation of failed components can not differ from the real state
        # "we cant unfail components"
        all_failed_components = failed_components.copy()
        while True:
            if len(failed_components) == 0:
                break
            print("Failed components: "+str(failed_components))
            new_failed_components = {}
            for component, errorId in failed_components.items():
                #print(component, errorId)
                probabilities = matrix.loc[component,errorId]
                #print("Probabilites: "+str(probabilities))
                for computed_component in Components.list():
                    if computed_component in all_failed_components:
                        continue
                    new_state = np.random.choice(ComponentFailure.list(), p=np.array(probabilities[computed_component])/np.sum(probabilities[computed_component]))
                    #print("New state: "+new_state)
                    if new_state == 'good':
                        continue
                    else:
                        all_failed_components[computed_component] = new_state
                        new_failed_components[computed_component] = new_state
            failed_components = new_failed_components
            
        return all_failed_components
        