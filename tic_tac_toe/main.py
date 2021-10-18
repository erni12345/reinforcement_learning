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
import pygame

from testing_strat import *





class TicTacToeEnv(Env):

    def __init__(self):
        self.action_space = Discrete(9)
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0]), high=np.array([0,0,0,0,0,0,0,0,0]))
        self.state = [0]*9


    def check_if_win_stoppped(self, action):

        checks = [(0,1,2),(3,4,5),(6,7,8),(0,4,8),(6,4,3),(0,3,6),(1,4,7),(2,5,8)]

        for x in checks:
            amount_of_2 = 0
            for y in x:
                if self.self.state[y] == 2:
                    amount_of_2 += 1
            if amount_of_2 >= 2 and action in x:

                return True
        
        return False


        
    def check_win(self):
        win_state = [
                    [self.state[0], self.state[1], self.state[2]],
                    [self.state[3], self.state[4], self.state[5]],
                    [self.state[6], self.state[7], self.state[8]],
                    [self.state[0], self.state[4], self.state[8]],
                    [self.state[6], self.state[4], self.state[2]],
                    [self.state[0], self.state[3], self.state[6]],
                    [self.state[1], self.state[4], self.state[7]],
                    [self.state[2], self.state[5], self.state[8]],
                    ]
        if [1, 1, 1] in win_state:
            return True, 1

        if [2,2,2] in win_state:
            return True, 2
        else:
            return False, 0
            
                

                    
    def step(self, action ): #player
        
        reward = -10

        if 0 not in self.state:
            print("draw!")
            reward += 150
            return self.state, reward, True, {}

        if self.state[action] == 2 or self.state[action] == 1:
            return self.state, -100, False, {}
        
        self.state[action] = 1
        
        empty_cells = [x for x in range(len(self.state)) if self.state[x] == 0 ]
        
        #Non AI does the move here
        possible_moves = [x for x in range(len(self.state)) if self.state[x] != 0 ]
        if len(empty_cells) >= 8:
            self.state[random.choice(possible_moves)] = 2
        elif possible_moves != []:
            state_grid = [[self.state[0], self.state[1], self.state[2]],[self.state[3], self.state[4], self.state[5]], [self.state[6], self.state[7], self.state[8]]]
            move = minimax(state_grid, abs(len(empty_cells)), 2)
            move_choice = (move[0]*3 + move[1])
            self.state[move_choice] = 2


        someone_one = self.check_win()[0]
        who_won = self.check_win()[1]

        
            
        if not someone_one:
            reward += 30
            return self.state, reward, False, {}

        if someone_one:
            if who_won == 1:
                reward += 150
                return self.state, reward, True, {}
            elif who_won == 2:
                reward -= 1000
                return self.state, reward, True, {}
        

    


    def render(self):
        
        board = f"""
        \r 
            {self.self.state[0]}, {self.self.state[1]}, {self.self.state[2]}
            {self.self.state[3]}, {self.self.state[4]}, {self.self.state[5]}
            {self.self.state[6]}, {self.self.state[7]}, {self.self.state[8]}
        \r
        """

        

        pos = {0:(500, 300),1:(610, 300),2:(720, 300),3:(500, 410),4:(610, 410),5:(720,410),6:(500,520),7:(610,520),8:(720,520)}
        for x in range(9):

                if self.self.state[x] == 0:
                    pygame.draw.rect(fenster, (255,255,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                elif self.self.state[x] == 1:
                    pygame.draw.rect(fenster, (0,0,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                else:
                    pygame.draw.rect(fenster, (255,0,0), pygame.Rect(pos[x][0], pos[x][1], 100, 100))

        pygame.display.update()

    def reset(self):
        self.state = [0] * 9
        self.done = False
        return self.state

    def _get_obs(self):
        return tuple(self.board)

    


"""
clock = pygame.time.Clock()
fenster = pygame.display.set_mode([1280, 720])
fenster.fill((100,100,100))

"""


def player_action(click):
    x, y = click
    check = {0:(500, 300),1:(610, 300),2:(720, 300),3:(500, 410),4:(610, 410),5:(720,410),6:(500,520),7:(610,520),8:(720,520)}

    for i in check:
        if check[i][0] < x < check[i][0] + 110 and check[i][1] < y < check[i][1]+110:
            return i
    
    return -1






env = TicTacToeEnv()
log_path = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO_New_Rules")
eval_callback = EvalCallback(env, eval_freq = 10000, best_model_save_path=PPO_PATH, verbose=1)
model = PPO.load(PPO_PATH, env)#PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
model.learn(total_timesteps=2000000, callback = eval_callback)

"""running = True
import time
obs = env.reset()
player_chose = False
done = False
while running:
    clock.tick(5)
    env.render()
    
    if not done:
        if player_chose:
            time.sleep(1)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action, 1)
            env.render()
            player_chose = False
        else:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN :
                    pos = pygame.mouse.get_pos()
                    player_move = player_action(pos)
                    obs, reward, done, info = env.step(player_move,2)
                    player_chose = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

"""













































        

        


        

