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

class GameEnv2048(Env):

    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=2048, shape=(4,4))
        self.state = [[0]*4 for x in range(4)]
        self.time = 0
        self.old_reward = 0
    
    def get_empty_cells(self):

        empty_cells = []

        for x in range(4):
            for y in range(4):
                if self.state[x][y] == 0:
                    empty_cells.append((x, y))

        return empty_cells

    def add_new_2(self):

        empty_cells = self.get_empty_cells()
        if empty_cells != []:
            choice = random.choice(empty_cells)

            self.state[choice[0]][choice[1]] = 2

    def loss(self):
    
        for i in range(4):
            for j in range(4):
                if(self.state[i][j] == 0):
                    return False

        for i in range(3):
            for j in range(3):
                if(self.state[i][j]== self.state[i + 1][j] or self.state[i][j]== self.state[i][j + 1]):
                    return False
 
        for j in range(3):
            if(self.state[3][j]== self.state[3][j + 1]):
                return False
 
        for i in range(3):
            if(self.state[i][3]== self.state[i + 1][3]):
                return False

        return True


    def transpose(self, grid):
        new_mat = []
        for i in range(4):
            new_mat.append([])
            for j in range(4):
                new_mat[i].append(grid[j][i])
        return new_mat


    def reverse(self, grid):
        new_mat =[]
        for i in range(4):
            new_mat.append([])
            for j in range(4):
                new_mat[i].append(grid[i][3 - j])
        return new_mat

    def merge(self, grid):
         
        changed = False
        
        for i in range(4):
            for j in range(3):
    
                # if current cell has same value as
                # next cell in the row and they
                # are non empty then
                if(grid[i][j] == grid[i][j + 1] and grid[i][j] != 0):
    
                    # double current cell value and
                    # empty the next cell
                    grid[i][j] = grid[i][j] * 2
                    grid[i][j + 1] = 0
    
                    # make bool variable True indicating
                    # the new grid after merging is
                    # different.
                    changed = True
    
        return self.state, changed

    
    def compress(self, grid):
     
        # bool variable to determine
        # any change happened or not
        changed = False
    
        # empty grid
        new_mat = []
    
        # with all cells empty
        for i in range(4):
            new_mat.append([0] * 4)
            
        # here we will shift entries
        # of each cell to it's extreme
        # left row by row
        # loop to traverse rows
        for i in range(4):
            pos = 0
    
            # loop to traverse each column
            # in respective row
            for j in range(4):
                if(grid[i][j] != 0):
                    
                    # if cell is non empty then
                    # we will shift it's number to
                    # previous empty cell in that row
                    # denoted by pos variable
                    new_mat[i][pos] = grid[i][j]
                    
                    if(j != pos):
                        changed = True
                    pos += 1
    
        # returning new compressed matrix
        # and the flag variable.
        return new_mat, changed

    
    def move_left(self, grid):
    
        # first compress the grid
        new_grid, changed1 = self.compress(grid)
    
        # then merge the cells.
        new_grid, changed2 = self.merge(new_grid)
        
        changed = changed1 or changed2
    
        # again compress after merging.
        new_grid, temp = self.compress(new_grid)
    
        # return new matrix and bool changed
        # telling whether the grid is same
        # or different
        return new_grid, changed
    
    # function to update the matrix
    # if we move / swipe right
    def move_right(self, grid):
    
        # to move right we just reverse
        # the matrix
        new_grid = self.reverse(grid)
    
        # then move left
        new_grid, changed = self.move_left(new_grid)
    
        # then again reverse matrix will
        # give us desired result
        new_grid = self.reverse(new_grid)
        return new_grid, changed
    
    # function to update the matrix
    # if we move / swipe up
    def move_up(self, grid):
    
        # to move up we just take
        # transpose of matrix
        new_grid = self.transpose(grid)
    
        # then move left (calling all
        # included functions) then
        new_grid, changed = self.move_left(new_grid)
    
        # again take transpose will give
        # desired results
        new_grid = self.transpose(new_grid)
        return new_grid, changed

    # function to update the matrix
    # if we move / swipe down
    def move_down(self, grid):
    
        # to move down we take transpose
        new_grid = self.transpose(grid)
    
        # move right and then again
        new_grid, changed = self.move_right(new_grid)
    
        # take transpose will give desired
        # results.
        new_grid = self.transpose(new_grid)
        return new_grid, changed
    

    def reward(self):
        
        reward = 0
        for x in self.state:
            reward += sum(x)
        return reward - self.old_reward

    def step(self, action):

        action_dict = {0:self.move_right(self.state),1:self.move_left(self.state),2:self.move_down(self.state),3:self.move_up(self.state) }

        if not self.loss():
            new_grid, changed = action_dict[action] # takes action
            if not changed:
                return self.state, -10, False, {}
            self.state = new_grid
            self.add_new_2()
            self.old_reward = self.reward()
            return self.state, self.old_reward, False, {}

        if self.loss():
            return self.state, self.reward(), True, {}


    def reset(self):
        self.state = [[0]*4 for x in range(4)]
        self.old_reward = 0
        self.add_new_2()

        return self.state

    def render(self):
        for x in self.state:
            print(x)


env = GameEnv2048()
log_path = os.path.join("Training", "Logs")
save_path = os.path.join("Training", "Saved Models", "V_1")
model = PPO.load(save_path)#PPO("MlpPolicy", env, verbose = 1, tensorboard_log=log_path, learning_rate = 0.0001)
"""model.learn(10000)
model.save(save_path)"""

done = False
obs = env.reset()
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    print("________________________")


