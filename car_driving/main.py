
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

env_name = "CarRacing-v0"
env = gym.make(env_name)
env = DummyVecEnv([lambda:env])
PPO_Path = os.path.join("Training", "Models", "PPO")
log_path = os.path.join("Training", "Logs")



for x in range(1000):
    model = PPO.load(PPO_Path, env)
    model.learn(total_timesteps=1000)
    model.save(PPO_Path)
    evaluate_policy(model,  env, n_eval_episodes = 1, render = True)

    
    

    
    





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