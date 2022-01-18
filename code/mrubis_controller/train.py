from agent.models.fix_predictor import FixPredictor
from agent.models.ranking_predictor import RankingPredictor
from agent.agent import Agent
from trainer.trainer import Trainer


if __name__ == "__main__":
    fix_predictor = FixPredictor()
    ranking_predictor = RankingPredictor()
    agent = Agent(fix_predictor, ranking_predictor)
    trainer = Trainer(agent=agent)
    trainer.train()