import gym
import numpy as np


class CMDPWrapper(gym.Wrapper):
    def __init__(self, env, cost_per_step_threshold=0.025,
                 terminate_on_violate=False, reward_shaping_on_violate=False):
        super(CMDPWrapper, self).__init__(env=env)
        self.terminate_on_violate = terminate_on_violate
        self.reward_shaping_on_violate = reward_shaping_on_violate
        self.cost_per_step_threshold = cost_per_step_threshold
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.env.observation_space.shape[0] + 1,),
                                                dtype=self.env.observation_space.dtype)

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        self.length = 0
        self.curr_cost = 0.
        return np.insert(self.obs, 0, 0.)

    def get_obs(self):
        return np.insert(self.obs, 0, self.curr_cost / self.length)

    def step(self, action):
        self.obs, reward, done, info = self.env.step(action)
        self.curr_cost += info['cost']
        self.length += 1
        if self.curr_cost > self.cost_per_step_threshold * self.length:
            if self.terminate_on_violate:
                done = True
            if self.reward_shaping_on_violate:
                reward = -info['cost']
        return self.get_obs(), reward, done, info
