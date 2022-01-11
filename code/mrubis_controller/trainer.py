class Trainer():
    def __init__(self, environment, agent, digital_twin):
        # Get instances of agent and digital twin
        #self.agent = agent
        #self.digital_twin = digital_twin
        #self.environment = environment
        pass

    def train(self):
        # gets data and calls train_agent and train_digital_twin

        #Loop:
            # train twin
            # train agent / or use exploration agent / use batch (both on real environment)

            # swtich to twin
            #Loop:
                # periodically execute "validation step" (short switch to the real environment) to detect if we can stay on the twin
                # break if we overfit
        pass

    def train_agent(self, data):
        # -------
        #   Optimzation step:
        # LOOP:
            # Predict fix & utility for shop
            # Send fix for each shop to environment (feedback: did we hit the right component?)
        # pass utilities to ranking model
        # predict ranking
        # send ranking to mRubis
        # calculate loss and optimize models with reward
        # -------
        pass

    def train_digital_twin(self, data):
        pass

    def observation_to_vector(self):
        pass

    def vector_to_observation(self):
        pass
