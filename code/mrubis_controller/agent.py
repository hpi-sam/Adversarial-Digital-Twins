class Agent():
    def __init__(self, fix_predictor, ranking_predictor):
        self.fix_predictor = fix_predictor
        self.ranking_predictor = ranking_predictor

    def predict_fix(self, shop_observation):
        # predict next fix for shop 
        # predict utility 
        return self.fix_predictor.forward(shop_observation)
    
    def predict_ranking(self, utilities):
        # predict an optimal ranking for predicted fixes
        return self.ranking_predictor.forward(utilities)

