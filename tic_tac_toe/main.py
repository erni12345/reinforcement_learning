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



clock = pygame.time.Clock()
fenster = pygame.display.set_mode([1280, 720])
fenster.fill((100,100,100))

class TicTacToeEnv(Env):

    """
     Envirnoement Tic Tac Toe
    """
    def __init__(self):
        """
            Initialisation des espaces d'observation et d'actn pour L'IA
        """
        self.action_space = Discrete(9)
        self.observation_space = Box(low=np.array([0,0,0,0,0,0,0,0,0]), high=np.array([0,0,0,0,0,0,0,0,0]))
        self.state = [0]*9
        self.time = 5


    def check_if_win_stoppped(self, action):
        """
        Verification de si la nouvelle action a arreter une vicotire

        Args:
            action (int): place joue

        Returns:
            [Bool]: [True si une victoire arreter, False si non]
        """

        checks = [(0,1,2),(3,4,5),(6,7,8),(0,4,8),(6,4,3),(0,3,6),(1,4,7),(2,5,8)]

        for x in checks:
            amount_of_2 = 0
            for y in x:
                if self.state[y] == 2:
                    amount_of_2 += 1
            if amount_of_2 >= 2 and action in x:

                return True
        
        return False


        
    def check_win(self):

        """
        Verifie le tableu, si 3 se suivent alors victoire
            

        Returns:
            [Boolean, Int]: [Si victoire et le gagnant]
        """
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
            
                

    def isValid(self, action):
        """Verifie si un mouvement est valide

        Args:
            action (Int): Case jouee

        Returns:
            Bool: True si Mouvement est autorise et False si non
        """
        if self.state[action] == 0:
            return True

        return False





    def reward(self, action):
        """
        Fonction reward qui calcule le reward donne a chaque mouvement.
        Verification si le mouvement est valide, si vicotire arrete, si victoire ou perte ou matche nul
        + Nombre de couts joue (empeche de tourner a jamais)

        Args:
            action (Int): Case jouee

        Returns:
            Int, Bool : Reward et Si fin de partie ou pas
        """
        reward = -10

        """if self.time <= 0:
            reward -= 1000
            return reward, True"""
        if self.check_if_win_stoppped(action):
            reward += 110
        if self.check_win()[1] == 1:
            reward += 110
            return reward, True
        elif self.check_win()[1] == 2:
            reward -= 1000
            return reward, True

        if 0 not in self.state and self.check_win[1] == 0:
            reward += 210
            return reward, True


        return reward, False



    def step(self, action, player ): #player
        """
        
        Fonction utilise a chaque etape

        Args:
            action (Int): Case jouee

        Returns:
             List, Int, Bool, Dict: State, Reward, Done, Info
        """
        self.time -= 1
        reward = self.reward(action)
        
        #Check for illegal Move
        """if not self.isValid(action):
            return self.state, -1000, True, {}"""


        #AI move
        self.state[action] = player


        return self.state, reward[0], reward[1], {}

        """empty_cells = [x for x in range(len(self.state)) if self.state[x] == 0 ]
        #Non AI does the move here
        possible_moves = [x for x in range(len(self.state)) if self.state[x] != 0 ]
        if len(empty_cells) >= 8:
            self.state[random.choice(possible_moves)] = 2
        elif possible_moves != []:
            state_grid = [[self.state[0], self.state[1], self.state[2]],[self.state[3], self.state[4], self.state[5]], [self.state[6], self.state[7], self.state[8]]]
            move = minimax(state_grid, abs(len(empty_cells)), 2)
            move_choice = (move[0]*3 + move[1])
            self.state[move_choice] = 2"""

        



    


    def render(self):
        
        board = f"""
        \r 
            {self.state[0]}, {self.state[1]}, {self.state[2]}
            {self.state[3]}, {self.state[4]}, {self.state[5]}
            {self.state[6]}, {self.state[7]}, {self.state[8]}
        \r
        """

        

        pos = {0:(500, 300),1:(610, 300),2:(720, 300),3:(500, 410),4:(610, 410),5:(720,410),6:(500,520),7:(610,520),8:(720,520)}
        for x in range(9):

                if self.state[x] == 0:
                    pygame.draw.rect(fenster, (255,255,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                elif self.state[x] == 1:
                    pygame.draw.rect(fenster, (0,0,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                else:
                    pygame.draw.rect(fenster, (255,0,0), pygame.Rect(pos[x][0], pos[x][1], 100, 100))

        pygame.display.update()

    def reset(self):
        self.state = [0] * 9
        self.done = False
        self.time = 5
        return self.state

    def _get_obs(self):
        return tuple(self.board)


def player_action(click):
    x, y = click
    check = {0:(500, 300),1:(610, 300),2:(720, 300),3:(500, 410),4:(610, 410),5:(720,410),6:(500,520),7:(610,520),8:(720,520)}

    for i in check:
        if check[i][0] < x < check[i][0] + 110 and check[i][1] < y < check[i][1]+110:
            return i
    
    return -1








env = TicTacToeEnv()
log_path = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO_New_Check_With_Time")
save_path = os.path.join("Training", "Saved Models", "best_model")
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 50, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=save_path, verbose=1)
model = PPO.load(save_path, env) #PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path) #
"""""model.learn(total_timesteps=200000, callback = eval_callback)
model.save(PPO_PATH)"""














































        

        


        

