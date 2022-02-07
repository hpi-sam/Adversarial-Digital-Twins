from agent.models.fix_predictor import FixPredictor
from agent.models.ranking_predictor import RankingPredictor
from agent.agent import Agent
from digital_twin.digital_twin import DigitalTwin, ShopDigitalTwin
from trainer.trainer import Trainer
import torch
import numpy
import random

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)
    fix_predictor = FixPredictor()
    ranking_predictor = RankingPredictor()
    agent = Agent(fix_predictor, ranking_predictor)
    # torch.save(agent.fix_predictor.state_dict(), 'modelweights.pth')
    agent.fix_predictor.load_state_dict(torch.load('modelweights.pth'))
    trainer = Trainer(agent=agent, digital_twin=DigitalTwin())
    trainer.train(max_runs=10000)
