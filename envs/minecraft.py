from .base_envs.slippery_grid import SlipperyGrid
from functools import partial
import numpy as np
from minigrid.core.world_object import Floor, Key
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from scipy.stats.kde import gaussian_kde
import PIL.Image as Image
from numpy import uint8
from minigrid.core.world_object import *
import os
from collections import defaultdict

OBJECT_TO_IDX.update({'ghost': len(OBJECT_TO_IDX)})
for i in range(4):
    OBJECT_TO_IDX.update({'coin_%s' % i: len(OBJECT_TO_IDX)})
SIZE = 50

class Coin(WorldObj):
    def __init__(self, class_name, color, background):
        super().__init__(class_name, color)
        self.bg = background

    def can_overlap(self):
        return True

    def render(self, img):
        if self.bg: self.bg.render(img)
        coin = Image.open(os.path.join(os.getcwd(), 'envs/base_envs/layouts/pacman/coin.png')).resize([SIZE, SIZE])
        im = Image.fromarray(img).convert('RGBA')
        im.putalpha(127)
        im.paste(coin, (25, 25), coin)

        img[:] = np.array(im.convert('RGB'))
    
    def toggle_on(self, env, pos):
        env.grid.set(*pos, self)
    
    def toggle_off(self, env, pos):
        env.grid.set(*pos, self.bg)


class Minecraft(SlipperyGrid):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)


        # define the labellings
        labels = np.empty([self.shape[0], self.shape[1]], dtype=object)
        labels[0:10, 0:10] = 'safe'
        labels[2, 7:10] = 'obstacle'
        labels[5, 0:3] = 'obstacle'
        labels[2][0] = 'wood'
        labels[4][9] = 'work_bench'
        # labels[2][4] = labels[9][0] = labels[7][7] = 'tool_shed'
        labels[0][7] = 'gold'

        # Used for tracking the agent for visualization purposes
        self.coins = {}
        self.rho_alphabet = ["obstacle", "grass", "wood", "work_bench", "gold"]
        self.rho_locations = defaultdict(list)
        self.labels = labels
        self.add_random_rewards()
        for col in range(len(self.labels)):
            for row, label in enumerate(self.labels[col]):
                if label == 'safe': 
                    self.coins[(row, col)] = Coin('coin_0', 'green', None)
                    continue

                if label == 'obstacle':
                    X = Floor('red')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_1', 'green', X)
                elif label == 'wood':
                    X = Floor('purple')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_1', 'green', X)
                elif label == 'grass':
                    X = Floor('green')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_1', 'green', X)
                # if label == 'iron':
                #     minecraft.grid.set(row, col, Floor('purple'))
                elif label == 'work_bench':
                    X = Floor('blue')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_2', 'green', X)
                # if label == 'tool_shed':
                #     minecraft.grid.set(row, col, Floor('blue'))
                elif label == 'gold':
                    X = Floor('yellow')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_3', 'green', X)
                self.rho_locations[label].append([row, col])

        # override the labels
        #self.labels
        self.cost = np.ones((shape[0], shape[1], 5))
        self.render_live = False
        self.render()
    
    def add_random_obstacle(self, num_obs=10):
        obs_locations = np.random.randint(10, size=(num_obs,2))
        for location in obs_locations:
            if self.labels[location[0]][location[1]] == 'safe':
                self.labels[location[0]][location[1]] = 'obstacle'
    
    def add_random_rewards(self, num_rews=10):
        num_added_rews = 0
        while num_added_rews < num_rews:
            rew_location = np.random.randint(10, size=(2))
            if self.labels[rew_location[0]][rew_location[1]] == 'safe':
                self.labels[rew_location[0]][rew_location[1]] = 'grass'
                num_added_rews += 1
                    
    
    def render(self, states = [], save_dir=None, mode='rgb_array', **kw):

        rows = []
        cols = []
        if states:
            dist = np.array([x for x in states])
            cols = dist[...,0].reshape(-1)
            rows = dist[...,1].reshape(-1)
            
            for row, col in zip(rows, cols):
                self.coins[row, col].toggle_on(self, [row, col])
        # import pdb; pdb.set_trace()
        img = super().render(mode, highlight=True, **kw)
        
        for row, col in zip(rows, cols):
            self.coins[row, col].toggle_off(self, [row, col])
        # self.window.fig.savefig("minecraft.png")
        import pdb; pdb.set_trace()
        return img
        
        # if save_dir is not None:



# create a SlipperyGrid object
shape = [10, 10]
minecraft = Minecraft(shape=shape, initial_state=[9, 5], slip_probability=0.)

