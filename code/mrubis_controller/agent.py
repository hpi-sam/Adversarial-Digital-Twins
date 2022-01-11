class Agent():
    def __init__(self, fix_predictor, ranking_predictor):
        self.fix_predictor = fix_predictor
        self.ranking_predictor = ranking_predictor

    def predict_fix(self, shop_obsrvation):
        # predict next fix for shop 
        # predict utility 
        pass
    
    def predict_ranking(self, utilities, issues):
        # predict an optimal ranking for predicted fixes
        pass

