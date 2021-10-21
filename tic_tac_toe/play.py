from main import *
import pygame
from numpy.lib.stride_tricks import DummyArray
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
from stable_baselines3.common.env_checker import check_env
import time


env = TicTacToeEnv()
log_path = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO_New_Rules", "best_model")
eval_callback = EvalCallback(env, eval_freq = 10000, best_model_save_path=PPO_PATH, verbose=1)
model = PPO.load(PPO_PATH, env)#PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
running = True
obs = env.reset()
PLAYER_TURN = False
AI_TURN = True
DONE = False





def player_action(click):
    x, y = click
    check = {0:(500, 300),1:(610, 300),2:(720, 300),3:(500, 410),4:(610, 410),5:(720,410),6:(500,520),7:(610,520),8:(720,520)}

    for i in check:
        if check[i][0] < x < check[i][0] + 110 and check[i][1] < y < check[i][1]+110:
            return i
    
    return -1




def ai_action(obs):

    move = model.predict(obs)
    

    return move






while running:

    
    clock.tick(5)
    env.render()
    if AI_TURN and not DONE:
        action, _ = ai_action(obs)
        print("AI ACTION : " + str(action))
        if env.isValid(action):
            
            obs, reward, DONE, info = env.step(action,1)
            AI_TURN = False
            PLAYER_TURN = True
        env.render()


    if PLAYER_TURN and not DONE:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN :
                    pos = pygame.mouse.get_pos()
                    player_move = player_action(pos)
                    obs, reward, DONE, info = env.step(player_move,2)
                    AI_TURN = True
                    PLAYER_TURN = False
                    env.render()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False