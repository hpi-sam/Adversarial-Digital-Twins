# Pytorch model
# Predict utility and next fix per shop
# start with single layer
import torch.nn as nn
import torch
import numpy
import random
class FixPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        random.seed(0)
        numpy.random.seed(0)
        self.input_dim = 20
        self.hidden_dim = 16
        self.batch_size = 1
        self.num_layers = 2

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        #self.utility_output = nn.Linear(self.hidden_dim * 2 * self.num_layers, 1)
        self.utility_output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 * self.num_layers, 1)
        )

        self.component_output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 * self.num_layers, 18),
            nn.Softmax(dim=0)
        )

        self.fix_output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 * self.num_layers, 5),
            nn.Softmax(dim=0)
        )


    def forward(self, shop_observation_vector, explore=False):
        if explore:
            return torch.randn(18), torch.randn(1), torch.randn(5)
        
        _, (hidden_values, _) = self.lstm(shop_observation_vector)
        predicted_component = self.component_output(hidden_values.view(self.hidden_dim*2*self.num_layers))

        predicted_utility_gain = self.utility_output(hidden_values.view(self.hidden_dim*2*self.num_layers))

        predicted_fix = self.fix_output(hidden_values.view(self.hidden_dim*2*self.num_layers))

        return predicted_component, predicted_utility_gain, predicted_fix
