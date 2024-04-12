import numpy as np
import gym
from gym import spaces


# Ego agent localisation
ego_home = np.array([0.0, 0.5])
#other_home = np.array([1.0, 0.5])


class Circle(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float32
        )
        self.action_space_other = spaces.Box(
            low = -np.pi,
            high = np.pi,
            shape = (1,),
            dtype = np.float32
        )

        self.radius = 1.0
        self.change_partner = 0.99
        self.reset_theta = 0.999
        self.ego = np.copy(ego_home)
        self.other = np.array([self.radius, 0.])
        #self.other_home = np.array([self.radius, 0.])
        self.theta = 0.0
        self.partner = 0
        self.timestep = 0


    def set_params(self, change_partner):
        self.change_partner = change_partner


    def _get_obs(self):
        return np.copy(self.ego)

    def _get_obs_other(self):
        return np.copy(self.other)


    def polar(self, theta):
        return self.radius * np.array([np.cos(theta), np.sin(theta)])


    def reset(self):
        return self._get_obs()


    def step(self, action):
        self.timestep += 1
        self.ego += action[0]
        #self.other = self.polar(action[1])
        reward = -np.linalg.norm(self.other - self.ego) * 100
        reward_other = -reward

        
        done = False
        if self.timestep == 10:
            self.timestep = 0
            #randomly reset the other agent
            #if np.random.random() > self.reset_theta:
             #    self.theta = np.random.uniform(0, 2*np.pi)
            # # choose a new partner from the options
            #if np.random.random() > self.change_partner:
                 #self.partner = np.random.choice(range(4))

            self.ego = np.copy(ego_home)
            self.other = self.polar(action[1])
        return [self._get_obs(),self._get_obs_other()], [reward, reward_other], done, {}

