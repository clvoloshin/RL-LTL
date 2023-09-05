from operator import add
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import io
from PIL import Image
from minigrid.minigrid_env import *

class SlipperyGrid(MiniGridEnv):
    """
    Slippery grid-world modelled as an MDP

    ...

    Attributes
    ----------
    shape: list
        1d list with two elements: 1st element is the num of row cells and the 2nd is the num of column cells (default [40, 40])
    initial_state : list
        1d list with two elements for initial state (default [0, 39])
    slip_probability: float
        probability of slipping (default 0.15)
    sink_states : list
        sinks states if any (default [])

    Methods
    -------
    reset()
        resets the MDP state
    step(action)
        changes the state of the MDP upon executing an action, where the action set is {right,up,left,down,stay}
    state_label(state)
        outputs the label of input state
    """

    def __init__(
            self,
            shape=None,
            initial_state=None,
            slip_probability=0.15,
            sink_states=None
    ):
        if sink_states is None:
            sink_states = []
        if shape is None:
            self.shape = [40, 40]
        else:
            self.shape = shape
        if initial_state is None:
            self.initial_state = [0, 39]
        else:
            self.initial_state = initial_state

        self.current_state = self.initial_state.copy()
        self.d0 = self.state_to_index(self.initial_state)
        self.slip_probability = slip_probability
        self.sink_states = sink_states
        self.labels = None

        # directional actions
        self.action_map = [
            "right",
            "up",
            "left",
            "down",
            "stay"
        ]
        mission_space = MissionSpace(mission_func=lambda : "Do the LTL task.",
                                         ordered_placeholders=None)
        super().__init__(
            mission_space=mission_space,
            width = max(self.shape[1], 3),
            height=max(self.shape[0],3),
            max_steps=1000,
            # Set this to True for maximum speed
            see_through_walls=True,
            render_mode="rgb_array"
        )
        super().reset()

        self.observation_space = gym.spaces.Discrete(shape[0] * shape[1])
        self.action_space = gym.spaces.Discrete(5)
        self.beta = 1-np.max(self.slip_probability) if isinstance(self.slip_probability, np.ndarray) else 1-self.slip_probability #min non-zero prob
        assert self.beta > 0

    def reset(self):
        self.current_state = self.initial_state.copy()
        return self.state_to_index(self.current_state), {}
    
    def state_to_index(self, state):
        return state[0] * self.shape[1] + state[1]
    
    def index_to_state(self, index):
        n_columns = self.shape[1]
        return [index // n_columns, index % n_columns]
    
    def T(self, state, action):
        state = self.index_to_state(state)
        slipperiness = self.slip_probability[state[0], state[1]] if isinstance(self.slip_probability, np.ndarray) else self.slip_probability

        p = np.zeros(self.observation_space.n)

        for tmp_action in self.action_map:
            # grid movement dynamics:
            if tmp_action == 'right':
                next_state = list(map(add, state, [0, 1]))
            elif tmp_action == 'up':
                next_state = list(map(add, state, [-1, 0]))
            elif tmp_action == 'left':
                next_state = list(map(add, state, [0, -1]))
            elif tmp_action == 'down':
                next_state = list(map(add, state, [1, 0]))
            elif tmp_action == 'stay':
                next_state = state

            # check for boundary violations
            if next_state[0] == self.shape[0]:
                next_state[0] = self.shape[0] - 1
            if next_state[1] == self.shape[1]:
                next_state[1] = self.shape[1] - 1
            if -1 in next_state:
                next_state[next_state.index(-1)] = 0
            
            if tmp_action == self.action_map[action]:
                p[self.state_to_index(next_state)] += 1-slipperiness
            
            p[self.state_to_index(next_state)] += slipperiness/len(self.action_map)

        assert np.isclose(sum(p), 1)
        return p

    def step(self, action_idx):
        
        slipperiness = self.slip_probability[self.current_state[0], self.current_state[1]] if isinstance(self.slip_probability, np.ndarray) else self.slip_probability
        
        # check if the agent is in a sink state
        if tuple(self.current_state) in self.sink_states:
            next_state = self.current_state
        else:
            # slipperiness
            if np.random.uniform() < slipperiness:
                action_idx = np.random.choice(self.action_space.n)

            action = self.action_map[action_idx]

            # grid movement dynamics:
            if action == 'right':
                next_state = list(map(add, self.current_state, [0, 1]))
            elif action == 'up':
                next_state = list(map(add, self.current_state, [-1, 0]))
            elif action == 'left':
                next_state = list(map(add, self.current_state, [0, -1]))
            elif action == 'down':
                next_state = list(map(add, self.current_state, [1, 0]))
            elif action == 'stay':
                next_state = self.current_state

            # check for boundary violations
            if next_state[0] == self.shape[0]:
                next_state[0] = self.shape[0] - 1
            if next_state[1] == self.shape[1]:
                next_state[1] = self.shape[1] - 1
            if -1 in next_state:
                next_state[next_state.index(-1)] = 0

            # check for obstacles
            if 'obstacle' in self.state_label(next_state):
                next_state = self.current_state

        # update current state
        reward = 1.0 if (next_state != self.current_state) else 0.0
        prev_state = self.current_state
        self.current_state = next_state

        next_state = self.state_to_index(next_state)
        # prev_state = self.state_to_index(prev_state)
        try:
            cost = self.cost[prev_state[0], prev_state[1], action_idx]
        except:
            import pdb; pdb.set_trace()
        done = False
        info = {'state': self.current_state}
        return next_state, 0, done, info
    
    def cost_shaping(self, prev_index, cur_index, action, automaton_movement, accepting_state_reached, rejecting_state_reached):
        
        cost = 1

        # if (automaton_movement and not rejecting_state_reached):
        #     cost = .5
        
        if accepting_state_reached:
            cost = 0
        
        if rejecting_state_reached:
            cost = 100

        return cost, False

    def get_cost(self, state, action):
        grid = self.index_to_state(state)
        return self.cost[grid[0], grid[1], action]
    
    def did_succeed(self, *args, **kw):
        return 0

    def label(self, state_idx):
        # labels = self.labels.get(self.state, None)
        state = self.index_to_state(state_idx)
        labels = [self.labels[state[0], state[1]]]
        if labels is None:
            return {}
        else:
            return {label:1 for label in labels}, {}
        
    def set_state(self, state):
        assert (state >= 0) and (state <= self.observation_space.n), 'Setting Environment to invalid state'
        self.current_state = self.index_to_state(state)
    
    def get_state(self):
        return self.state_to_index(self.current_state)

    def state_label(self, state):
        return self.labels[state[0], state[1]]
    
    # For rendering
    def _gen_grid(self, width, height):
        self.agent_pos = tuple(self.current_state[::-1])
        self.agent_dir = 0
        self.grid = Grid(width, height)
        self.mission = ''
        #import pdb; pdb.set_trace()
        # for row in range(len(labels)):
        #     for col, label in enumerate(labels[row]):
        #         if label == 'safe': continue
        #         self.grid.set(row, col, Floor())
    
    def render(self, mode='rgb_array', **kw):
        self.agent_pos = tuple(self.current_state[::-1])
        self.agent_dir = 0
        # print(self.agent_pos[::-1])
        return super().render()


            


        