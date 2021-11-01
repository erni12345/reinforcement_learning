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
new_gen_path = os.path.join("Training", "Saved Models", "new_gen")
trainer_path = os.path.join("Training", "Saved Models", "trainer")
vs_random = os.path.join("Training", "Saved Models", "vs_random")
vs_bot_save = os.path.join("Training", "Saved Models", "vs_bot.1.7", "best_model", "best_model")
vs_bot = os.path.join("Training", "Saved Models", "vs_bot_2_8", "best_model")

eval_callback = EvalCallback(env, eval_freq = 10000, best_model_save_path=PPO_PATH, verbose=1)
model = PPO.load(vs_bot, env)#PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
running = True
bot = PPO.load(vs_bot, env)
obs = env.reset()
PLAYER_TURN = False
AI_TURN = True
DONE = False





def player_action(click):
    x, y = click
    check = {0:(240, 300),1:(350, 300),2:(460, 300),3:(240, 410),4:(350, 410),5:(460,410),6:(240,520),7:(350,520),8:(460,520)}

    for i in check:
        if check[i][0] < x < check[i][0] + 110 and check[i][1] < y < check[i][1]+110:
            return i
    
    return -1




def ai_action(obs):

    move = model.predict(obs)
    

    return move




while running:

    if DONE:
        obs = env.reset()
        DONE = False
        AI_TURN = True
        PLAYER_TURN = False

    clock.tick(5)
    env.render()
    if AI_TURN and not DONE:
        action, _ = ai_action(obs)
        print("AI ACTION : " + str(action))
        if env.isValid(action):
            
            obs, reward, DONE, info = env.step(action,1)
            print(reward)
            AI_TURN = False
            PLAYER_TURN = True
        env.render()


    if PLAYER_TURN and not DONE:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN :
                    pos = pygame.mouse.get_pos()
                    new_state = []
                    for x in obs:
                        if x == 1:
                            new_state.append(2)
                        elif x == 2:
                            new_state.append(1)
                        else:
                            new_state.append(x)
                    
                    
                    player_move = player_action(pos)
                    obs, reward, DONE, info = env.step(player_move,2)
                    AI_TURN = True
                    PLAYER_TURN = False
                    env.render()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

"""

def see_probs_recur_other(board, depth, turn):

    if str(board) in memo_iz_second:
        return memo_iz_second[str(board)]


    if turn == 1:
        turn = 2
    else:
        turn = 1

    check = check_win_other(board)
    prob = 0
    

    if check[0]:
        return check[1]
    
    if depth == 0:
        return 0
    
    for x in empty_cells(board):
        board_copy = board[::]
        board_copy[x] = turn
        temp_prob = see_probs_recur_other(board_copy, depth-1, turn)
        prob += temp_prob

    memo_iz_second[str(board)] = prob/(depth)
    return prob/(depth)


def get_probs_other(state):
    proba_dict = {}
    for x in empty_cells(state):
        copy = state[::]
        copy[x] = 2
        proba_dict[x] = see_probs_recur_other(copy, len(empty_cells(copy)), 2)

    return proba_dict








def empty_cells(state):
    
    cells = []

    for x in range(9):
        if state[x] == 0:
            cells.append(x)

    return cells
        
def check_win(board):

    win_state = [
                [board[0], board[1], board[2]],
                [board[3], board[4], board[5]],
                [board[6], board[7], board[8]],
                [board[0], board[4], board[8]],
                [board[6], board[4], board[2]],
                [board[0], board[3], board[6]],
                [board[1], board[4], board[7]],
                [board[2], board[5], board[8]],
                ]
    if [1, 1, 1] in win_state:
        return True, 1

    if [2,2,2] in win_state:
        return True, -1
    if 0 not in board:
        return True, 0
    else:
        return False, 0
            

probs = []
memo_iz = {}
def get_probs(board):
    proba_dict = {}
    for x in empty_cells(board):
        copy = board[::]
        copy[x] = 1
        
        proba_dict[x] = see_probs_recur(copy, len(empty_cells(copy)), 1)

    return proba_dict

def see_probs_recur(board, depth, turn):

    
    if turn == 1:
        turn = 2
    else:
        turn = 1

    check = check_win(board)
    prob = 0
    

    if check[0]:
        return check[1]
    
    if depth == 1:
        board_copy = board[::]
        board_copy[empty_cells(board_copy)[0]] = turn
        return check_win(board_copy)[1]
    
    for x in empty_cells(board):
        board_copy = board[::]
        board_copy[x] = turn
        temp_prob = see_probs_recur(board_copy, depth-1, turn)
        prob += temp_prob
    
    return prob/(depth)






import time
start_time = time.time()
print(get_probs([0,0,0,0,0,0,0,0,0]))
print("--- %s seconds ---" % (time.time() - start_time))
"""
