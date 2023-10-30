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
        # labels[0:3, 5] = 'obstacle'
        labels[2, 7:10] = 'obstacle'
        # labels[4][5] = labels[8][1] = labels[8][7] = labels[9][9] = 'grass'
        # labels[3][2] = labels[7][3] = labels[5][7] = labels[0][0] = 'grass'
        # labels[2][2] = labels[2][3] = labels[0][8] = labels[4][8] = 'grass'
        # labels[2][1] = labels[3][1] = 'grass'
        # labels[0][3] = labels[4][0] = labels[6][8] = labels[9][4] = 'iron'
        labels[2][0] = labels[3][0] = 'wood'
        labels[4][9] = 'work_bench'
        # labels[2][4] = labels[9][0] = labels[7][7] = 'tool_shed'
        labels[0][7] = 'gold'

        # Used for tracking the agent for visualization purposes
        self.coins = {}
        
        for col in range(len(labels)):
            for row, label in enumerate(labels[col]):
                if label == 'safe': 
                    self.coins[(row, col)] = Coin('coin_0', 'green', None)
                    
                    continue

                if label == 'obstacle':
                    X = Floor('red')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_1', 'green', X)
                elif label == 'grass':
                    X = Floor('green')
                    self.grid.set(row, col, X)
                    self.coins[(row, col)] = Coin('coin_1', 'green', X)
                elif label == 'wood':
                    X = Floor('purple')
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

        # override the labels
        self.labels = labels
        self.cost = np.ones((shape[0], shape[1], 5))

        
    
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
        return img
        
        # if save_dir is not None:
        #     self.window.fig.savefig(save_dir)



# create a SlipperyGrid object
shape = [10, 10]
minecraft = Minecraft(shape=shape, initial_state=[9, 2], slip_probability=0.)





    
# def test(n_traj, sim, policy, J, c_min, map= lambda x: x):
#     traj = []
#     costs = []
#     random_actions = []
#     for n in range(n_traj):
#         costs.append([])
#         traj.append([sim.reset()])
#         random_actions.append([])
#         for t in range(100):
#             try:
#                 sim.mdp.render()
#             except:
#                 pass

#             try:
#                 action = policy[map(traj[-1][-1])]
#                 random_act = 0
#             except:
#                 # logger.warn('Policy sees new state: %d' % map(traj[-1][-1]))
#                 # import pdb; pdb.set_trace()
#                 random_act = 1
#                 action = np.random.choice(sim.mdp.action_space.n)

#             next_state, cost, _, info = sim.step(action)

#             traj[-1].append(next_state)
#             costs[-1].append(1)
#             random_actions[-1].append(random_act)
#             # if info['goal']: break
    
#     print([np.sum(cost) for cost in costs], np.mean([np.sum(cost) for cost in costs]), np.std([np.sum(cost) for cost in costs]))
#     return np.array([[[sim.mdp.index_to_state(sim.mapstate_idx][0]) for state_idx in T] for T in traj])

# minecraft.plot = partial(plot, minecraft)
# minecraft.test = test
