from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


class ShowerEnv(Env):

    def __init__(self):
        
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60

    def step(self, action):
        
        self.state += action - 1
        self.shower_length -= 1
        
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
        
        if self.shower_length <= 0:
            done = True

        else:
            done = False
        
        self.state += random.randint(-1,1)

        info = {}

        return self.state, reward, done, info


    def render(self):
        pass

    def reset(self):
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60 

        return self.state


env = ShowerEnv()

log_path = os.path.join("Training", "Logs")
DQN_Path = os.path.join("Training", "Saved Models", "DQN_Test")
PPO_Path = os.path.join("Training", "Saved Models", "PPO_Model_CartPole")
DQN_Path = os.path.join("Training", "Saved Models", "DQN_Model_CartPole", "best_model", "best_model")
model = DQN.load(DQN_Path, env = env)


stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 45, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=DQN_Path, verbose=1)

"""model.learn(total_timesteps = 2000000, callback = eval_callback)
model.save(DQN_Path)"""


episodes = 6

for x in range(1, episodes+16):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print(f"Episode : {x} Score : {score}")
