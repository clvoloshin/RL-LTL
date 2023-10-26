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
    def __init__(self, state_shp, num_cycles, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1
        self.num_cycles = num_cycles

        self.states = np.zeros((max_,) + state_shp)
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.ltl_rewards = np.zeros((max_, self.num_cycles))
        self.cycle_rewards = np.zeros((max_, self.num_cycles))
        self.next_states = np.zeros((max_,) + state_shp)
        self.buchis = np.array([0 for _ in range(max_)])
        self.next_buchis = np.array([0 for _ in range(max_)])

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, b, a, r, lr, cr, s_, b_):
        self.counter += 1
        self.states[self.counter % self.max_] = s
        self.buchis[self.counter % self.max_] = b
        self.next_states[self.counter % self.max_] = s_
        self.next_buchis[self.counter % self.max_] = b_
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.ltl_rewards[self.counter % self.max_] = lr
        self.cycle_rewards[self.counter % self.max_] = cr
        self.all_current_traj.append(self.counter % self.max_)
    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    def get_current_traj(self):
        idxs = np.array(self.current_traj)
        df = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_']

        idxs = np.array(self.all_current_traj)
        df2 = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df2.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_']
        return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]

    def sample(self, batchsize):
        if self.counter < batchsize:
            # import pdb; pdb.set_trace()
            return self.states, self.buchis, self.actions, self.rewards, self.ltl_rewards, self.cycle_rewards, self.next_states, self.next_buchis
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]

class DQN(nn.Module):
    def __init__(self, env_space, act_space, param):
        super(DQN, self).__init__()
        if len(env_space['mdp'].shape) == 0:
            envsize = 1
        else:
            envsize = env_space['mdp'].shape[0]
        
        self.actor = nn.Sequential(
                        nn.Linear(envsize, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, env_space['buchi'].n * act_space['total'])
                    )
        self.shp = (env_space['buchi'].n, act_space['total'])
            
        # Mask actions not available
        self.mask = torch.ones((env_space['buchi'].n, act_space['total'])).type(torch.bool).to(device)
        if act_space['total'] != act_space['mdp'].n:
            for buchi in range(env_space['buchi'].n):
                if buchi in act_space:
                    eps = 1 + act_space[buchi].n
                else:
                    eps = 1
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
        with torch.no_grad():
            if len(state.shape) == 0:
                state = state.unsqueeze(0).to(device)
            #mport pdb; pdb.set_trace()
            qs = torch.reshape(self.actor(state), self.shp)[buchi_state]
            masked_qs = torch.masked.MaskedTensor(qs, self.mask[buchi_state])
            act = int(masked_qs.argmax())
            is_eps = act >= self.n_mdp_actions
            return act, is_eps
    
    def random_act(self, state, buchi_state):
        X = self.mask[buchi_state].cpu().numpy()
        pos = np.random.choice(sum(X), size=1)
        idx = np.take(X.nonzero(), pos, axis=1)
        act = idx[0][0]
        is_eps = act >= self.n_mdp_actions
        return act, is_eps