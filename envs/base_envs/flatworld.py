import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import gym
import gym.spaces as spaces

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utls.plotutils import plotlive
fontsize = 24
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize) 

# # figure settings
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
sns.set_theme()

class FlatWorld(gym.Env):
    
    def __init__(
        self,
        continuous_actions=False,
        render_mode: Optional[str] = "human",
    ):
        
        low = np.array(
            [
                -2,
                2
            ]
        ).astype(np.float32)
        high = np.array(
            [
                -2,
                2
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        self.continuous_actions = continuous_actions
        if continuous_actions:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # up, right, down, left, nothing
            self.action_space = spaces.Discrete(5)
            

        # self.obs_1 = np.array([0.0, 0.9, -1.0, -0.5])     # red box in bottom right corner
        # self.obs_2 = np.array([.2, 0.7, 0.8, 1.2])        # green box in top right corner
        # self.obs_3 = np.array([0.0, 0.0, 0.4])            # blue circle in the center
        # self.obs_4 = np.array([-1.0, -0.7, -0.2, 0.5])    # orange box on the left

        # self.goal = np.array([1, 1, .2])
        self.obs_1 = np.array([.9/2, -1.5/2., .3])     # red box in bottom right corner
        self.obs_2 = np.array([.9/2, 1., .3])        # green box in top right corner
        self.obs_3 = np.array([0.0, 0.0, 0.8])            # blue circle in the center
        self.obs_4 = np.array([-1.7/2, .3/2, .3])    # orange box on the left
        
        # self.circles = [(self.obs_1, 'r'), (self.obs_2, 'g'), (self.obs_4, 'y'), (self.obs_3, 'b')]
        self.circles = [(self.obs_1, 'r'), (self.obs_4, 'y'), (self.obs_3, 'b')]

        self.state = np.array([-1, -1])
        self.render_mode = render_mode
        self.fig, self.ax = plt.subplots(1, 1)
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        
        self.state = np.array([-1, -1])

        return self.state, {}
    
    def get_state(self):
        return self.state
        
    def label(self, state):
        state
        signal, labels = {}, {}
        for circle, color in self.circles:
            val = np.linalg.norm(state - circle[:-1])
            if val < circle[-1]:
                labels.update({color:1})
                val = -1
            
            signal.update({color: -val})
        return labels, signal
    
    def did_succeed(self, x, a, r, x_, d, t):
        return int('goal' in self.label(x))

    def step(self, action):

        if not self.continuous_actions:
            if action == 0:
                u = np.array([[0, .5]])
            elif action == 1:
                u = np.array([[.5, 0]])
            elif action == 2:
                u = np.array([[0, -.5]])
            elif action == 3:
                u = np.array([[-.5, 0]])
            elif action == 4:
                u = np.array([[0, 0]])
            else:
                raise NotImplemented
        else:
            u = action
        
        Δt = .4
        A = np.eye(2)
        B = np.eye(2) * Δt
        # action = np.clip(u, -1, +1).astype(np.float32)
        action = u

        self.state = A @ self.state.reshape(2, 1) + B @ action.reshape(2, 1)
        self.state = self.state.reshape(-1)
        cost = np.linalg.norm(action)
        terminated = False

        return self.state, cost, terminated, {}

    @plotlive
    def render(self, states = [], save_dir=None):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # plot the environment given the obstacles
        # plt.figure(figsize=(10,10))
        for obs, color in self.circles:           
            # theta = np.linspace( 0 , 2 * np.pi , 150 )
 
            # radius = obs[2]
            
            # x = radius * np.cos( theta ) + obs[0]
            # y = radius * np.sin( theta ) + obs[1]
 
            # # x, y = [obs[0] + obs[2]*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs[1] + obs[2]*np.sin(t) for t in np.arange(0,3*np.pi,0.1)]
            # self.ax.plot(x, y, c=color, linewidth=2)

            patch = plt.Circle((obs[0], obs[1]), obs[2], color=color, fill=True, alpha=.2)
            # # x, y = [obs[0] + obs[2]*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs[1] + obs[2]*np.sin(t) for t in np.arange(0,3*np.pi,0.1)]
            # # self.ax.plot(x, y, c=color, linewidth=2)
            self.ax.add_patch(patch)
        
        # for obs, color in [(self.goal, 'orange')]:
        #     self.ax.plot([obs[0] + obs[2]*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs[1] + obs[2]*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c=color, linewidth=2)
        
        # plt.plot([self.obs_1[0], self.obs_1[0], self.obs_1[1], self.obs_1[1], self.obs_1[0]], [self.obs_1[2], self.obs_1[3], self.obs_1[3], self.obs_1[2], self.obs_1[2]], c="red", linewidth=5)
        # plt.plot([self.obs_2[0], self.obs_2[0], self.obs_2[1], self.obs_2[1], self.obs_2[0]], [self.obs_2[2], self.obs_2[3], self.obs_2[3], self.obs_2[2], self.obs_2[2]], c="green", linewidth=5)
        # plt.plot([self.obs_4[0], self.obs_4[0], self.obs_4[1], self.obs_4[1], self.obs_4[0]], [self.obs_4[2], self.obs_4[3], self.obs_4[3], self.obs_4[2], self.obs_4[2]], c="orange", linewidth=5)
        # plt.plot([self.obs_3[0] + self.obs_3[2]*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [self.obs_3[1] + self.obs_3[2]*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)

        # for state in states:
        #     self.ax.scatter([state[0]], [state[1]], s=100, marker='-', c="g")
        self.ax.plot(np.array(states)[:, 0], np.array(states)[:, 1], color='green', marker='o', linestyle='dashed',
            linewidth=2, markersize=4)

        self.ax.scatter([self.state[0]], [self.state[1]], s=100, marker='o', c="g")


        # self.ax.scatter([self.goal[0]], [self.goal[1]], s=20, marker='*', c="orange")
        self.ax.axis('square')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])

        if save_dir is not None:
            self.fig.savefig(save_dir)
        
        # self.ax.grid()
        # plt.show(block=False)