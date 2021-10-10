import os
import gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold



environment_name = "CartPole-v0"
env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])
log_path = os.path.join("Training", "Logs")
DQN_Path = os.path.join("Training", "Saved Models", "DQN_Test")

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
#model.learn(total_timesteps = 20000)
PPO_Path = os.path.join("Training", "Saved Models", "PPO_Model_CartPole")
#model.save(PPO_Path)

model = PPO.load(PPO_Path, env=env)

episodes = 6

for x in range(1, episodes+16):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print(f"Episode : {x} Score : {score}")


save_path = os.path.join("Training", "Saved Models")
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 200, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=save_path, verbose=1)
net_arch = [128, 128]
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
"""model.learn(total_timesteps=20000, callback = eval_callback)
model.save(DQN_Path)"""


