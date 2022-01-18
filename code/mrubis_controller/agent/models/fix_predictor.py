# Pytorch model
# Predict utility and next fix per shop
# start with single layer
import torch.nn as nn
import torch
class FixPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_part = nn.Sequential(
            nn.Linear(18*5, 64),
            nn.LeakyReLU()
        )
        self.utility_output = nn.Sequential(
            nn.Linear(64, 1)
        )
        self.fix_output = nn.Sequential(
            nn.Linear(64, 18*2),
            nn.Softmax()
        )


    def forward(self, shop_observation_vector):
        #shop_observation_vector = 2D matrix of length number of components, depth number of states, one 1 per column.
        # return fix, utility
        # fix is a 2D matrix with length number of components and depth number of possible fixes.
        # utility is a float

        # failure_index = np.where(shop_observation_vector[1:] == 1)
        # all_components_list = Components.list()
        # all_fixes_list = Fixes.list()
        # fix = np.zeros((len(all_fixes_list), len(all_components_list)))
        # fix[0, failure_index[1]] = 1
        print(shop_observation_vector)
        print(shop_observation_vector.shape)
        hidden_values = self.input_part(torch.flatten(torch.from_numpy(shop_observation_vector).float()))
        predicted_utility_gain = self.utility_output(hidden_values)
        predicted_fix = torch.reshape(self.fix_output(hidden_values), (2, 18))

        return predicted_fix, predicted_utility_gain
