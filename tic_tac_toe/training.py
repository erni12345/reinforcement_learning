from typing import cast
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
from stable_baselines3.common.vec_env import VecFrameStack


#from testing_strat import *



memo_iz_first = {}
memo_iz_second = {}


clock = pygame.time.Clock()
fenster = pygame.display.set_mode([800, 800])
fenster.fill((0,0,0))

X_Image = pygame.image.load("X.png").convert_alpha()
O_Image = pygame.image.load("O.png").convert_alpha()




def empty_cells(state):
        """
        Each empty cell will be added into cells' list
        :param state: the state of the current board
        :return: a list of empty cells
        """
        cells = []

        for x in range(9):
            if state[x] == 0:
                cells.append(x)

        return cells



def check_win(board):

        """
        Verifie le tableu, si 3 se suivent alors victoire
            

        Returns:
            [Boolean, Int]: [Si victoire et le gagnant]
        """
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


def check_win_other(board):

        """
        Verifie le tableu, si 3 se suivent alors victoire
            

        Returns:
            [Boolean, Int]: [Si victoire et le gagnant]
        """
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
            return True, -1

        if [2,2,2] in win_state:
            return True, 1
        if 0 not in board:
            return True, 0
        else:
            return False, 0

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
        self.time = 4

    def see_probs_recur(self, board, depth, turn):

        if str(board) in memo_iz_first:
            return memo_iz_first[str(board)]


        if turn == 1:
            turn = 2
        else:
            turn = 1

        check = check_win(board)
        prob = 0
        

        if check[0]:
            return check[1]
        
        if depth ==  0:
            return 0
        
        for x in empty_cells(board):
            board_copy = board[::]
            board_copy[x] = turn
            temp_prob = self.see_probs_recur(board_copy, depth-1, turn)
            prob += temp_prob

        memo_iz_first[str(board)] = prob/(depth)
        return prob/(depth)


    def get_probs(self, state):
        proba_dict = {}
        for x in empty_cells(state):
            copy = state[::]
            copy[x] = 1
            proba_dict[x] = self.see_probs_recur(copy, len(empty_cells(copy)), 1)

        return proba_dict

    
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



    def check_win_in_class(self):

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
        if 0 not in self.state:
            return True, 0
        else:
            return False, 0


    def reward(self):
        """
        Fonction reward qui calcule le reward donne a chaque mouvement.
        Verification si le mouvement est valide, si vicotire arrete, si victoire ou perte ou matche nul
        + Nombre de couts joue (empeche de tourner a jamais)

        Args:
            action (Int): Case jouee

        Returns:
            Int, Bool : Reward et Si fin de partie ou pas
        """
        reward = 0

        check = self.check_win_in_class()
        if check[1] == 2:
            reward -= 1
            return reward, True
        elif check[1] == 1:
            reward += 1
            return reward, True

        elif check[0] and check[1] == 0:
            return 0, True

        if self.time <= 0:
            reward -= 1
            return reward, True
        

        return 0, False


    def make_best_move(self):

        probs = get_probs_other(self.state)
        maxx = -100
        best = -1
    
        for x in probs:
            if probs[x] >= maxx:
                maxx = probs[x]
                best = x
        return best

    def made_the_best_move(self, probs, action):
        play = -1
        max_play = -1
        for x in probs:
            if probs[x] >= max_play:
                play = x
                max_play = probs[x]
        
        if action == play:
            return (True, max_play)

        return (False, -1)

    def step(self, action, player): #player
        """
        
        Fonction utilise a chaque etape

        Args:
            action (Int): Case jouee

        Returns:
             List, Int, Bool, Dict: State, Reward, Done, Info
        """
        
        
        #Check for illegal Move

        if self.reward()[1]:
            return self.state, self.reward()[0], self.reward()[1], {}
        if not self.isValid(action):
            return self.state, -1, True, {}

        self.time -= 0 #1

        #AI move (THAT IS BEING TRAINED)
        copy = self.state[::]
        self.state[action] = player #1
        reward = self.reward()

        """if not reward[1]:
            win_prob_dict = self.get_probs(copy)
            made_best = self.made_the_best_move(win_prob_dict, action)
            if made_best[0]:
                reward = (made_best[1], False)"""
        if reward[1]:
            return self.state, reward[0], reward[1], {}

        """#AI move that isnt being trained
        if self.vs_cpu:
            new_state = []
            for x in self.state:
                if x == 1:
                    new_state.append(2)
                elif x == 2:
                    new_state.append(1)
                else:
                    new_state.append(x)
            
            bot_action, _ = bot.predict(new_state)
            if self.state[bot_action] == 0:
                self.state[bot_action] = 2
            else:
                self.time += 1
                self.state[action] = 0

        elif self.vs_cpu == "random":
            bot_action = random.choice(empty_cells(self.state))
            self.state[bot_action] = 2
        else:
            bot_action = self.make_best_move()
            if self.state[bot_action] == 0:
                self.state[bot_action] = 2
            else:
                self.time += 1
                self.state[action] = 0

        reward = self.reward()"""

        return self.state, reward[0], reward[1], {}



    


    def render(self):

        """
        Fonction render, prend en charge l'interface du jeu. Mise a jours des mouvements et elements. 
        """
        pos = {0:(240, 300),1:(350, 300),2:(460, 300),3:(240, 410),4:(350, 410),5:(460,410),6:(240,520),7:(350,520),8:(460,520)}
        for x in range(9):

                if self.state[x] == 0:
                    pass
                elif self.state[x] == 1:
                    pygame.draw.rect(fenster, (255,255,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                    fenster.blit(X_Image, (pos[x][0], pos[x][1]))
                else:
                    pygame.draw.rect(fenster, (255,255,255), pygame.Rect(pos[x][0], pos[x][1], 100, 100))
                    fenster.blit(O_Image, (pos[x][0], pos[x][1]))


        pygame.display.update()

    def reset(self):
        """Recommence le Jeu

        Returns:
            state: matrice du jeu
        """
        fenster.fill((0,0,0))
        pygame.draw.rect(fenster, (255,255,255), pygame.Rect(340,300, 10, 330))
        pygame.draw.rect(fenster, (255,255,255), pygame.Rect(450, 300, 10, 330))
        pygame.draw.rect(fenster, (255,255,255), pygame.Rect(240, 400, 330, 10))
        pygame.draw.rect(fenster, (255,255,255), pygame.Rect(240, 510, 330, 10))
        self.state = [0] * 9
        self.done = False
        self.time = 5
        """self.vs_cpu = random.choice([True, False, False, False, "random", "random"])
        should = random.randint(0,1)
        if should == 1:
            self.time = 4
            self.state[random.randint(0,8)] = 2"""
        return self.state

    def _get_obs(self):
        return tuple(self.board)






def player_action(click):
    """Fonction qui determine dans quelle case l'utilisateur a touche

    Args:
        click (tuple): (x,y) position

    Returns:
        int: lieu
    """
    x, y = click
    check = {0:(240, 300),1:(350, 300),2:(460, 300),3:(240, 410),4:(350, 410),5:(460,410),6:(240,520),7:(350,520),8:(460,520)}

    for i in check:
        if check[i][0] < x < check[i][0] + 110 and check[i][1] < y < check[i][1]+110:
            return i
    
    return -1




"""log_path = os.path.join("Training", "Logs")
vs_random = os.path.join("Training", "Saved Models", "vs_random")
vs_bot = os.path.join("Training", "Saved Models", "vs_bot_2_7_2", "best_model")
vs_bot_save_best = os.path.join("Training", "Saved Models", "vs_bot_3")

env = TicTacToeEnv()
bot = PPO.load(vs_bot, env)
model = PPO.load(vs_bot, env, learning_rate = 0.0001) #("MlpPolicy", env, verbose = 1, tensorboard_log=log_path, learning_rate = 0.0001)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 1.5, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=vs_bot_save_best, verbose=1)

model.learn(1000000, callback = eval_callback)"""








"""
model = PPO.load(new_gen_path, env)
model.learn(100)"""






































        

        


        

