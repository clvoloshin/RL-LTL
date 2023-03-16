import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import numpy.ma as ma
import pandas as pd

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class Buffer:
    def __init__(self, state_shp, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1

        self.states = np.zeros((max_,) + state_shp)
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.next_states = np.zeros((max_,) + state_shp)
        self.buchis = np.array([0 for _ in range(max_)])
        self.next_buchis = np.array([0 for _ in range(max_)])

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, b, a, r, s_, b_):
        self.counter += 1
        self.states[self.counter % self.max_] = s
        self.buchis[self.counter % self.max_] = b
        self.next_states[self.counter % self.max_] = s_
        self.next_buchis[self.counter % self.max_] = b_
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.all_current_traj.append(self.counter % self.max_)
    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    def get_current_traj(self):
        idxs = np.array(self.current_traj)
        df = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df.columns = ['s', 'b', 'a', 'r', 's_', 'b_']

        idxs = np.array(self.all_current_traj)
        df2 = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df2.columns = ['s', 'b', 'a', 'r', 's_', 'b_']
        return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]

    def sample(self, batchsize):
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]

class BufferStandard:
    def __init__(self, state_shp, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1

        self.states = np.zeros((max_,) + state_shp)
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.next_states = np.zeros((max_,) + state_shp)

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, a, r, s_,):
        self.counter += 1
        self.states[self.counter % self.max_] = s
        self.next_states[self.counter % self.max_] = s_
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.all_current_traj.append(self.counter % self.max_)
    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    def get_current_traj(self):
        idxs = np.array(self.current_traj)
        df = pd.DataFrame([self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs]]).T
        df.columns = ['s', 'a', 'r', 's_']

        idxs = np.array(self.all_current_traj)
        df2 = pd.DataFrame([self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs]]).T
        df2.columns = ['s', 'a', 'r', 's_',]
        return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs]

    def sample(self, batchsize):
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs]

class DQNSTL(nn.Module):
    def __init__(self, env_space, act_space, param, num_heads):
        super(DQNSTL, self).__init__()
        
        self.actor_base = nn.Sequential(
                        nn.Linear(env_space['mdp'].shape[0], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64)
                    )
        self.heads = [nn.Linear(64, act_space['mdp']) for _ in range(num_heads)]

        self.gamma = param['gamma']
        self.n_mdp_actions = act_space['mdp'].n
        
    def forward_base(self, state):
        return self.actor_base(state).squeeze()
    
    def forward_head(self, inp, head_idx):
        return self.heads[head_idx](inp).squeeze()
    
    def act(self, state):
        #TODO: double check - the top operator as a head should be for the whole formula?
        q_intermed = self.forward_base(state).squeeze()
        qs = self.forward_head(q_intermed, 0)
        act = int(qs.argmax())
        return act
    
    def random_act(self, state):
        act = np.random.choice(self.n_mdp_actions, size=1)
        return act 

class DQN(nn.Module):
    def __init__(self, env_space, act_space, param):
        super(DQN, self).__init__()
        
        self.actor = nn.Sequential(
                        nn.Linear(env_space['mdp'].shape[0], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, env_space['buchi'].n * act_space['total'])
                    )
        self.shp = (env_space['buchi'].n, act_space['total'])
            
        # Mask actions not available
        self.mask = torch.ones((env_space['buchi'].n, act_space['total'])).type(torch.bool)
        if act_space['total'] != act_space['mdp'].n:
            for buchi in range(env_space['buchi'].n):
                try:
                    eps = act_space['total'] - 1 + act_space[buchi].n
                except:
                    eps = act_space['total'] - 1
                self.mask[buchi, eps:] = False

        self.gamma = param['gamma']
        self.n_mdp_actions = act_space['mdp'].n
        
    def forward(self, state, buchi, to_mask=True):
        all_qs = torch.reshape(self.actor(state), (-1,) + self.shp)
        qs = torch.take_along_dim(all_qs, buchi, dim=1).squeeze()
        if to_mask:
            out = torch.masked.MaskedTensor(qs, self.mask[buchi].squeeze())
        else:
            out = qs
        return out
    
    def act(self, state, buchi_state):
        qs = torch.reshape(self.actor(state), self.shp)[buchi_state]
        masked_qs = torch.masked.MaskedTensor(qs, self.mask[buchi_state])
        act = int(masked_qs.argmax())
        is_eps = act >= self.n_mdp_actions
        return act, is_eps, 0
    
    def random_act(self, state, buchi_state):
        X = self.mask[buchi_state].numpy()
        pos = np.random.choice(sum(X), size=1)
        idx = np.take(X.nonzero(), pos, axis=1)
        act = idx[0][0]
        is_eps = act >= self.n_mdp_actions
        return act, is_eps, 0
        