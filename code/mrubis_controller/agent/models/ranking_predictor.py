# Initial idea: sort by utility

class RankingPredictor():
    def __init__(self):
        pass

    def forward(self, utilities):
        #TODO: Implement actual model
        # utilities is a vector of length number of shops with utilities for each shop
        # return array of shop indices with array[0] being the top priority.
        return [index for _, index in sorted(zip(utilities, list(range(len(utilities)))), reverse=True)]