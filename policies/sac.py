import torch
import torch.nn as nn
import math
import os
import numpy as np
import pandas as pd
import numpy.ma as ma
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal

# Adapted from https://github.com/pranz24/pytorch-soft-actor-critic

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class Buffer:
    def __init__(self, state_shp, action_shp, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1

        self.states = np.zeros((max_,) + state_shp)
        self.actions = np.zeros((max_,) + action_shp)
        self.rewards = np.array([0 for _ in range(max_)])
        self.ltl_rewards = np.array([0 for _ in range(max_)])
        self.constrained_rewards = np.array([0 for _ in range(max_)])
        self.next_states = np.zeros((max_,) + state_shp)
        self.buchis = np.array([0 for _ in range(max_)])
        self.next_buchis = np.array([0 for _ in range(max_)])
        self.terminals = np.array([0 for _ in range(max_)])

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, b, a, r, lr, cr, s_, b_, terminal):
        self.counter += 1
        # if r > 0: import pdb; pdb.set_trace()
        self.states[self.counter % self.max_] = s
        self.buchis[self.counter % self.max_] = b
        self.next_states[self.counter % self.max_] = s_
        self.next_buchis[self.counter % self.max_] = b_
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.ltl_rewards[self.counter % self.max_] = lr
        self.constrained_rewards[self.counter % self.max_] = cr
        self.terminals[self.counter % self.max_] = terminal
        self.all_current_traj.append(self.counter % self.max_)

    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    def get_current_traj(self):
        idxs = np.array(self.current_traj)
        df = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.constrained_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs], self.terminals[idxs]]).T
        df.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_', 't']

        idxs = np.array(self.all_current_traj)
        df2 = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.constrained_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs], self.terminals[idxs]]).T
        df2.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_', 't']
        return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.constrained_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs], self.terminals[idxs]

    def sample(self, batchsize):
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.constrained_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs], self.terminals[idxs]


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, env_space, param):
        super(ValueNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(env_space['mdp'].shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, env_space['buchi'].n)
            )

    def forward(self, state, buchi, to_mask=True):
        all_qs = torch.reshape(self.actor(state), (-1,) + self.shp)
        qs = torch.take_along_dim(all_qs, buchi, dim=1).squeeze()
        if to_mask:
            out = torch.masked.MaskedTensor(qs, self.mask[buchi].squeeze())
        else:
            out = qs
        return out


class QNetwork(nn.Module):
    def __init__(self, env_space, act_space, param):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
                        nn.Linear(env_space['mdp'].shape[0] + act_space['mdp'].shape[0], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, env_space['buchi'].n)
                    )
        # Q2 architecture
        self.q2 = nn.Sequential(
                        nn.Linear(env_space['mdp'].shape[0] + act_space['mdp'].shape[0], 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, env_space['buchi'].n)
                    )
        self.shp = (env_space['buchi'].n, act_space['total'])
            
        # Mask actions not available
        # self.mask = torch.ones((env_space['buchi'].n, act_space['total'])).type(torch.bool)
        # if act_space['total'] > 1:
        #     for buchi in range(env_space['buchi'].n):
        #         try:
        #             eps = act_space['total'] - 1 + env_space['buchi'].n
        #         except:
        #             eps = act_space['total'] - 1
        #         self.mask[buchi, eps:] = False    

        self.gamma = param['gamma']

    def forward_one(self, state, buchi, q=1): #, to_mask=True):
        model = self.q1 if q == 1 else self.q2
        #import pdb; pdb.set_trace()
        all_qs = torch.reshape(model(state), (-1,) + self.shp)
        qs = torch.take_along_dim(all_qs, buchi, dim=1).squeeze()
        # if to_mask:
        #     out = torch.masked.MaskedTensor(qs, self.mask[buchi].squeeze())
        # else:
        #     out = qs
        return qs
    
    def forward(self, state, buchi, action, to_mask=True):
        xu = torch.cat([state, action], 1)
        q1 = self.forward_one(xu, buchi, q=1)#, to_mask=to_mask)
        q2 = self.forward_one(xu, buchi, q=2)# to_mask=to_mask)
        return q1, q2


class GaussianPolicy(nn.Module):
    def __init__(self, env_space, act_space, param):
        super(GaussianPolicy, self).__init__()
        
        has_continuous_action_space = True
        self.has_continuous_action_space = has_continuous_action_space        
        if has_continuous_action_space:
            self.action_dim = act_space['mdp'].shape[0]
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(env_space['mdp'].shape[0], 64),
                            nn.ReLU(), #relu for other experiments
                            nn.Linear(64, 64),
                            nn.ReLU(),
                        )
            
            self.mean_head = nn.Sequential(
                            nn.Linear(64, env_space['buchi'].n * self.action_dim),
                            nn.Tanh()
                        )
            self.log_std_head = nn.Sequential(
                            nn.Linear(64, env_space['buchi'].n * self.action_dim),
                            nn.Tanh()
                        )
            self.action_switch = nn.Linear(64, env_space['buchi'].n * act_space['total']) # for buchi epsilons

            with torch.no_grad():
                # bias towards no epsilons in the beginning
                self.action_switch.bias[::act_space['total']] = 5.

                # bias towards left turn
                # self.main_head[0].bias[::self.action_dim] = 0
                                        

            self.main_shp = (env_space['buchi'].n, self.action_dim)
            self.shp = (env_space['buchi'].n, act_space['total'])
            
            # Mask actions not available
            self.mask = torch.ones((env_space['buchi'].n, act_space['total'])).type(torch.bool)
            if act_space['total'] > 1:
                for buchi in range(env_space['buchi'].n):
                    try:
                        eps = act_space['total'] - 1 + act_space[buchi].n
                    except:
                        eps = act_space['total'] - 1
                    self.mask[buchi, eps:] = False                        
        else:
            raise NotImplemented
        # action rescaling

        # self.action_scale = torch.FloatTensor(
        #     (action_space.high - action_space.low) / 2.)
        # self.action_bias = torch.FloatTensor(
        #     (action_space.high + action_space.low) / 2.)

    def masked_softmax(self, vec, mask, dim=1, tol=1e-7):
        float_mask = mask.float().to(device)
        masked_vec = vec * float_mask
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * float_mask
        masked_exps += tol # make sure you dont get -inf when log
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, state, buchi_state):
        body = self.actor(state)
        action_mean_head = self.mean_head(body)
        action_log_std_head = self.log_std_head(body)
        if state.shape[0] > 1:
            action_mean = torch.reshape(action_mean_head, (-1,) + self.main_shp)
            action_log_std = torch.reshape(action_log_std_head, (-1,) + self.main_shp)
            action_mean = torch.take_along_dim(action_mean, buchi_state, dim=1).squeeze()
            action_log_std = torch.take_along_dim(action_log_std, buchi_state, dim=1).squeeze()
            action_switch_head_all = torch.reshape(self.action_switch(body), (-1,) + self.shp)
            action_switch = action_switch_head_all
            mask = self.mask.unsqueeze(0).repeat(len(action_switch), 1, 1).to(device)
            probs_all = self.masked_softmax(action_switch, mask, -1)
            probs = torch.take_along_dim(probs_all, buchi_state, dim=1).squeeze()
        else:
            action_mean = torch.reshape(action_mean_head, self.main_shp)[buchi_state]
            action_log_std = torch.reshape(action_log_std_head, self.main_shp)[buchi_state]
            action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
            action_switch = action_switch_head_all
            mask = self.mask

            ### Bias against epsilon actions by multiplication by +10
            # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
            probs = self.masked_softmax(action_switch, mask, -1)[buchi_state]

        # action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
        # masked_head_all = torch.masked.MaskedTensor(action_switch_head_all, self.mask)
        # probs_all = F.softmax(masked_head_all)
        # probs = probs_all.to_tensor(0)
        
        # action_switch = action_switch_head_all#[buchi_state]
        # mask = self.mask#[buchi_state]

        # ### Bias against epsilon actions by multiplication by +10
        # # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
        # probs = self.masked_softmax(action_switch, mask, -1)

        return action_mean, action_log_std, probs

    def sample(self, state, buchi_state):
        is_eps = False
        mean, log_std, probs = self.forward(state, buchi_state)
        std = log_std.exp()
        # try:
        #     action_or_eps = Categorical(probs)
        # except:
        #     import pdb; pdb.set_trace()
        if state.shape[0] > 1: #TODO: resolve dimensionality mismatches with epsilon actions
            action_or_eps = Categorical(probs.unsqueeze(-1))
            #cov_mat = torch.diag_embed(std).to(device)
            act_or_eps = action_or_eps.sample()
            normal = Normal(mean, std)  # change to multivariate normal
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            #import pdb; pdb.set_trace()
            action = x_t
            log_prob = normal.log_prob(x_t) + torch.tile(action_or_eps.log_prob(act_or_eps).unsqueeze(-1), (1, x_t.shape[-1])) # prob_of_non_eps * prob of your actual action
            log_prob = log_prob.sum(1, keepdim=True)
            clipped_action = torch.clip(action, -1, 1)
            action = clipped_action
            return action, log_prob, mean, act_or_eps  # TODO: figure out how to take the act_or_eps action if it is prescribed
        else:
            action_or_eps = Categorical(probs)
            act_or_eps = action_or_eps.sample()
            if act_or_eps == 0:
                #cov_mat = torch.diag_embed(std).to(device)
                normal = Normal(mean, std)
                x_t = normal.sample()  # for reparameterization trick (mean + std * N(0,1))
                #y_t = torch.tanh(x_t)
                action = x_t #* self.action_scale + self.action_bias
                log_prob = normal.log_prob(x_t) + torch.tile(action_or_eps.log_prob(act_or_eps), x_t.shape)

                clipped_action = torch.clip(action, -1, 1)
                action = clipped_action

            else:
                is_eps = True
                action = act_or_eps
                log_prob = action_or_eps.log_prob(act_or_eps)

        # Enforcing Action Bound
        #log_prob = log_prob.sum(1, keepdim=True)
        #mean = torch.tanh(mean) #* self.action_scale + self.action_bias
        return action, log_prob, mean, is_eps

    def to(self, device):
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
