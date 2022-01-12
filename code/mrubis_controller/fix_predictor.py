# Pytorch model
# Predict utility and next fix per shop
# start with single layer
import numpy as np
from failure_propagator.components import Components
from failure_propagator.fixes import Fixes
class FixPredictor():
    def __init__(self):
        pass

    def forward(self, shop_observation_vector):
        #TODO: Implement actual model
        #shop_observation_vector = 2D matrix of length number of components, depth number of states, one 1 per column.
        # return fix, utility
        # fix is a 2D matrix with length number of components and depth number of possible fixes.
        # utility is a float
        failure_index = np.where(shop_observation_vector[1:] == 1)
        all_components_list = Components.list()
        all_fixes_list = Fixes.list()
        fix = np.zeros((len(all_fixes_list), len(all_components_list)))
        fix[0, failure_index[1]] = 1
        return fix, 0
