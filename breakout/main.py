import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import atari_py
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


environment_name = "Breakout-v0"
env = gym.make(environment_name)

"""episodes = 6
for x in range(1, episodes+16):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward

    print(f"Episode : {x} Score : {score}")"""


log_path = os.path.join("Training", "Logs")
A2C_path = os.path.join("Training", "Saved Models", "A2C")
save_path = os.path.join("Training", "Saved Models", "best")

env = make_atari_env(environment_name, n_envs = 4, seed = 0)
env = VecFrameStack(env, n_stack=4)

model = A2C.load(A2C_path, env)               #('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 45, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=save_path, verbose=1)
evaluate_policy(model, env, n_eval_episodes = 20, render = True)
"""model.learn(total_timesteps=10000000)
model.save(A2C_path)"""
