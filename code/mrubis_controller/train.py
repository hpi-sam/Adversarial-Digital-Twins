import wandb
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
    #trainer = Trainer(agent=agent, digital_twin=DigitalTwin())
    torch.save(agent.fix_predictor.state_dict(), 'modelweights.pth')
    for train_duration in (100, 5000,):# (10, 50, 100, 150, 200, 300, 500, 1000):
        agent.fix_predictor.load_state_dict(torch.load('modelweights.pth'))
        trainer = Trainer(agent=agent, digital_twin=DigitalTwin())
        trainer.train(max_runs=5000, num_synchronization=train_duration)
        trainer.environment.close_socket()
        wandb.finish()
        input("Restart mRUBiS and hit enter to continue")
