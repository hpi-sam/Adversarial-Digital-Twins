import json
import gym
from gym import spaces
from failure_propagator.failure_propagator import FailureProgagator


class MRubisEnvironment(gym.Env):
    def __init__(self, host='localhost', port=8080):
        super(MRubisEnvironment,  self).__init__()
        self.environment = FailureProgagator(
            host=host, port=port, json_path='path.json')
        self.number_of_shops = 0
        self.number_of_issues_per_shop = {}
        self.mrubis_state = {}

    def step(self, action):
        # action has same type as order tuples
        self.environment.send_order_in_which_to_apply_fixes(action)

        observation = self.environment.get_from_mrubis(
            message=json.dumps(
                {shop_name: [issue_component_tuple[1] for issue_component_tuple in issue_component_tuples]
                 for shop_name, issue_component_tuples in self.components_fixed_in_this_run.items()}
            )
        )

        # TODO: parse reward (utility) from observation for each shop
        # TODO: should the env know when to be done? count the runs here

        # return observation, reward, done, info
        return observation, 0, False, {}

    def reset(self):
        '''Query mRUBiS for the number of shops, get their initial states'''
        self.number_of_shops = self.environment.get_number_of_shops()
        #logger.info(f'Number of mRUBIS shops: {self.number_of_shops}')
        for _ in range(self.number_of_shops):
            shop_state = self.environment.get_initial_state()
            shop_name = next(iter(shop_state))
            self.mrubis_state[shop_name] = shop_state[shop_name]
        return self.mrubis_state

    def close(self):
        pass


# if __name__ == "__main__":
#     env = MRubisEnvironment()
#     state = env.reset()
#     print(state)
