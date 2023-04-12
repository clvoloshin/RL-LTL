from abc import ABCMeta, abstractmethod
import gym
import gym.spaces
import numpy as np
from copy import deepcopy
import time

def timeit(f):
    def wrapped(*args, **kw):
        tic = time.time()
        result = f(*args, **kw)
        toc = time.time()
        print('{} {} {} ms'.format('Sim', f.__name__, (toc - tic) * 1000))
        return result
    
    return wrapped

class AbstractEnv(metaclass=ABCMeta):
    @abstractmethod
    def label(state):
        return
    
    @abstractmethod
    def reset(self):
        return
    
    @abstractmethod
    def step(self, action):
        return
    
    def from_sparse_to_full(self, D):
        T = np.zeros((self.observation_space.n, self.action_space.n+1, self.observation_space.n))
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n + 1):
                for next_state in range(self.observation_space.n):
                    try:
                        T[state][action][next_state] = self.T[state][action][next_state]
                    except:
                        pass
                
                    # if action == self.action_space.n:
                    #     T[state][action][-1] = 1
        
        C = []
        for state in range(self.observation_space.n):
            C.append([0]*(self.action_space.n + 1))
            for action in range(self.action_space.n + 1):
                try:
                    C[state][action] = self.cost[state][action]
                except:
                    C[state][action] = D

        return T, C

class Simulator(gym.Env):
    def __init__(self, mdp, automaton):
        self.mdp = mdp
        self.automaton = automaton
        spaces = {
            'mdp': self.mdp.observation_space,
            'buchi': gym.spaces.Discrete(self.automaton.n_states)
            }
        self.observation_space = spaces #self.mdp.observation_space.shape[0] + self.automaton.n_states

        eps_per_state = [len(val) for key,val in self.automaton.automaton.eps.items()] + [0]
       
        spaces = {
            'mdp': self.mdp.action_space,
        }
        spaces.update({key: gym.spaces.Discrete(len(val)) for key, val in self.automaton.automaton.eps.items()})
        self.action_space = dict(spaces)
        
        try:
            self.action_space['total'] = sum([val.n for key, val in self.action_space.items()])
        except:
            self.action_space['total'] = 1 + max(eps_per_state)
        
        self.accepting_states = set()
        self.rejecting_states = set()
        self.inf_often = []
    
    def unnormalize(self, states):
        try:
            return self.mdp.unnormalize(states)
        except:
            return states
    
    def did_succeed(self, x, a, r, x_, d, t):
        return self.mdp.did_succeed(x['mdp'], a, r, x_['mdp'], d, t)
        # return self.mdp.did_succeed(x[:self.mdp.observation_space.shape[0]], a, r, x_[:self.mdp.observation_space.shape[0]], d, t)
    
    def label(self, state):
        raise NotImplemented
        # return self.mdp.label(state[:self.mdp.observation_space.shape[0]])        
            
    def reset(self, make_aut_init_state_random=False):
        try:
            #allow reset at any point, even if using Monitor
            self.mdp.stats_recorder.done = True
        except:
            pass

        state, _ = self.mdp.reset()
        label, _ = self.mdp.label(state)
        
        self.automaton.reset()
        if make_aut_init_state_random:
            self.automaton.set_state(np.random.choice(self.automaton.n_states - 1))
        automaton_state = self.automaton.step(label)

        return {'mdp': state, 'buchi': automaton_state}, {}

        # one_hot = np.zeros(self.automaton.n_states)
        # one_hot[automaton_state] = 1.
        # return np.hstack([state, one_hot]), {}
    
    # @timeit
    # def simulate(self, state, action):

    #     env_state, aut_state = state
    #     self.mdp.set_state(env_state)
    #     self.automaton.set_state(aut_state)
    #     output = self.step(action)
        
    #     new_output = ((output[0], self.mdp.get_state(), self.automaton.get_state()), output[1], output[2], output[3])
        
    #     self.mdp.set_state(env_state)
    #     self.automaton.set_state(aut_state)
        
    #     return new_output

    def next_buchi(self, mdp_state, desired_current_aut_state, eps_action=None):
        if eps_action is None:
            current_aut_state = self.automaton.get_state()
            label, _ = self.mdp.label(mdp_state)
            self.automaton.set_state(desired_current_aut_state)
            automaton_state = self.automaton.step(label)
            self.automaton.set_state(current_aut_state)
            # if (automaton_state in self.automaton.automaton.accepting_states) or (automaton_state != current_aut_state):
            if (automaton_state in self.automaton.automaton.accepting_states):
                accepting_rejecting_neutral = 1
            elif automaton_state == (self.automaton.automaton.n_states - 1):
                accepting_rejecting_neutral = -1
            else:
                accepting_rejecting_neutral = 0
            return automaton_state, accepting_rejecting_neutral
        else:
            current_aut_state = self.automaton.get_state()
            label, _ = self.mdp.label(mdp_state)
            self.automaton.set_state(desired_current_aut_state)
            automaton_state = self.automaton.epsilon_step(eps_action)
            automaton_state = self.automaton.step(label) # Do we take this step right now???
            self.automaton.set_state(current_aut_state)
            # if (automaton_state in self.automaton.automaton.accepting_states) or (automaton_state != current_aut_state):
            if (automaton_state in self.automaton.automaton.accepting_states):
                accepting_rejecting_neutral = 1
            elif automaton_state == (self.automaton.automaton.n_states - 1):
                accepting_rejecting_neutral = -1
            else:
                accepting_rejecting_neutral = 0
            return automaton_state, accepting_rejecting_neutral
    
    # @timeit
    def step(self, action, is_eps=False):
        current_mdp_state, current_aut_state = self.mdp.get_state(), self.automaton.get_state()

        if is_eps: 
            #epsilon transition
            state = self.mdp.get_state()
            label, _ = self.mdp.label(state)
            try:
                automaton_state = self.automaton.epsilon_step(action - self.mdp.action_space.n) # discrete
            except:
                automaton_state = self.automaton.epsilon_step(action - 1) # continuous

            automaton_state = self.automaton.step(label) # Do we take this step right now???
            cost = 0
            done = False
            info = self.mdp.get_info()
            # dic = {'state' : self.mdp.index_to_state(self.mdp.get_state())}
            # info = {'mdp_state': state, 'aut_state': automaton_state}
            # next_state = self.states.setdefault((state, automaton_state), len(self.states))
            # self.map.setdefault(self.states[(state, automaton_state)], (state, automaton_state))
        else:
            output = self.mdp.step(action)
            try:
                state, cost, done, _, info = output 
            except:
                state, cost, done, info = output
            label, _ = self.mdp.label(state)
            automaton_state = self.automaton.step(label)
        

        new_info = {'prev_mdp_state': current_mdp_state, 'prev_aut_state': current_aut_state , 's_': state, 'aut_state': automaton_state, 'label': label, 'is_accepting': automaton_state in self.automaton.automaton.accepting_states}
        try:
            new_info.update(info)
        except:
            import pdb; pdb.set_trace()

        if automaton_state is None:
            import pdb; pdb.set_trace()

        # if automaton_state in self.automaton.automaton.accepting_states: # accepting state
        #     self.accepting_states.add(state)
        # elif automaton_state == (self.automaton.automaton.n_states - 1): # rejecting state
        #     self.rejecting_states.add(state)

        # info.update({'fail': automaton_state in self.rejecting_states, 'goal': automaton_state in self.accepting_states})
        
        # if simulate:
        #     self.mdp.set_state(current_mdp_state)
        #     self.automaton.set_state(current_aut_state)

        return {'mdp': state, 'buchi': automaton_state}, cost, done, new_info
        # one_hot = np.zeros(self.automaton.n_states)
        # one_hot[automaton_state] = 1.
        # return np.hstack([state, one_hot]), cost, done, info
    
    def render(self, *args, **kw):
        return self.mdp.render(*args, **kw)
    
    def plot(self, *args, **kwargs):
        try:
            self.mdp.plot(*args, **kwargs)
        except:
            print('Cannot Print Policy')
    
    def test(self, *args, **kwargs):
        try:
            return self.mdp.test(*args, **kwargs)
        except:
            print('Cannot test Policy')
