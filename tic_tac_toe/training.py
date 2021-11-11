"""

Ficher pour entrainer le model d'intelligence Artficielle

Utilisation de l'algorithm PPO pour entrainer

"""



#importations
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




memo_iz_first = {}
memo_iz_second = {}



def empty_cells(state):
    """Function qui renvoie la liste de case libres

    Args:
        state (list): matrice du jeu

    Returns:
        list: liste de cases libres
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
    Verification pour joueur 2 (utilise pour entrainer IA contre ancienne version d'elle meme)

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

    """Function recursive qui calcul la propabilite de victoire d'un mouvement
       Utilise pour aller contre l'ordinateur

       utilise la programation dynamique afin de ne pas refaire d'etapes

    Returns:
        float : prob de gagner [0;1]
    """

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
    """Fonction qui renvoie la probabilite de gagner de chaque mouvement possible
        Pour aller contre ordinateur
    Args:
        state (list): matrice qui represent le tableau 

    Returns:
        dict: posibilite et pourcentage de victoire
    """
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
        """Function recursive qui calcul la propabilite de victoire d'un mouvement

            utilise la programation dynamique afin de ne pas refaire d'etapes

            Returns:
                float : prob de gagner [0;1]
        """
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
        """Fonction qui renvoie la probabilite de gagner de chaque mouvement possible
        Args:
            state (list): matrice qui represent le tableau 

        Returns:
            dict: posibilite et pourcentage de victoire
        """
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
        """
        Foction qui calcul le meilleur mouvement

        Returns:
            Int : case ou jouer
        """

        probs = get_probs_other(self.state)
        maxx = -100
        best = -1
    
        for x in probs:
            if probs[x] >= maxx:
                maxx = probs[x]
                best = x
        return best

    def made_the_best_move(self, probs, action):
        """fonction qui verifie si le meilleur mouvement a etait fait

        Args:
            probs (Dict): dictionaires de mouvements et probabilites
            action (Int): mouvement jouee par l'IA

        Returns:
            Bool, float : vrai ou faux si meilelr mouvement, prob de ganger
        """
        play = -1
        max_play = -1
        for x in probs:
            if probs[x] >= max_play:
                play = x
                max_play = probs[x]
        
        if action == play:
            return (True, max_play)

        return (False, -1)

    def step(self, action): #player
        """
        
        Fonction utilise a chaque etape

        1. Verifcation du mouvement
        2. Calcul du "reward" pour le mouvement fait
        3. Faire le mouvement + verification si victoire
        4. Calcul du mouvement de l'adverssaire
        5. Verification de victoire

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

        self.time -= 1

        #AI move (THAT IS BEING TRAINED)
        copy = self.state[::]
        self.state[action] = 1
        reward = self.reward()
        win = reward[1]
        reward = reward[0]
        if not win:
            win_prob_dict = self.get_probs(copy)
            reward = win_prob_dict[action]
    
        if win:
            return self.state, reward, win, {}

        #AI move that isnt being trained
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

        reward_cpu = self.reward()

        return self.state, reward, reward_cpu[1], {}


    def reset(self):
        """Recommence le Jeu
            Determine si l'IA commence premier ou deuxieme et s'il joue contre un ancienne version,
            ou le meiller mouvement ou un contre aleatoire

        Returns:
            state: matrice du jeu
        """
        self.state = [0] * 9
        self.done = False
        self.time = 5
        self.vs_cpu = random.choice([True, False, False, False, "random", "random"])
        should = random.randint(0,1)
        if should == 1:
            self.time = 4
            self.state[random.randint(0,8)] = 2
        return self.state

    def _get_obs(self):
        return tuple(self.board)







"""
Entrainement
"""
log_path = os.path.join("Training", "Logs")
vs_random = os.path.join("Training", "Saved Models", "vs_random")
vs_bot = os.path.join("Training", "Saved Models", "vs_bot_2_8", "best_model")
vs_bot_save = os.path.join("Training", "Saved Models", "vs_bot_3_2")
vs_bot_savee = os.path.join("Training", "Saved Models", "vs_bot_3_3")
vs_bot_save_best = os.path.join("Training", "Saved Models", "vs_bot_3_1_best_model")

env = TicTacToeEnv()
bot = PPO.load(vs_bot_save, env)
model = PPO.load(vs_bot_save, env, learning_rate = 0.0001) #("MlpPolicy", env, verbose = 1, tensorboard_log=log_path, learning_rate = 0.0001)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 1.5, verbose = 1)
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, eval_freq = 10000, best_model_save_path=vs_bot_save_best, verbose=1)

model.learn(1000000)
model.save(vs_bot_savee)












































        

        


        

