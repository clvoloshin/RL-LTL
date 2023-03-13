import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import time
torch.autograd.set_detect_anomaly(True)

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
# torch.default_device(device)

class Trajectory:
    def __init__(self, action_placeholder) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.act_idxs = []
        self.is_eps = []
        self.logprobs = []
        self.counter = 0
        self.done = False
        self.has_reward = False
        self.action_placeholder = action_placeholder # should be of MDP action shape
    
    def add(self, s, a, r, s_, logprob):
        self.counter += 1
        # if r > 0: import pdb; pdb.set_trace()
        self.states.append(s)
        self.next_states.append(s_)
        self.actions.append(a)
        self.rewards.append(r if r >= 0 else 0)
        self.has_reward = self.has_reward or (r > 0)
        self.done = self.done or (r < 0)
        self.logprobs.append(logprob)
    
    def copy(self):
        return deepcopy(self)
        

class RolloutBuffer:
    def __init__(self, state_shp, action_shp, max_ = 1000, to_hallucinate=False) -> None:
        self.trajectories = []
        self.all_reward_trajectories = []
        self.all_no_reward_trajectories = []
        self.first_action_was_epsilon = False
        self.action_placeholder = np.zeros(action_shp)
        self.max_ = max_
        self.to_hallucinate = to_hallucinate
        self.main_trajectory = None
    
    def add_experience(self, s, a, r, s_, logprobs):
        
        if len(self.trajectories) == 0: 
                traj = Trajectory(self.action_placeholder)
                self.trajectories.append(traj)
        
        traj = self.trajectories[-1]
        traj.add(s, a, r, s_, logprobs)
    
    def get_normalized_discounted_rewards(self, gamma):

        all_rewards = []
        for traj in self.trajectories:
            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward in reversed(traj.rewards):
                discounted_reward = reward + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            all_rewards += rewards # extend list
                        
        # Normalizing the rewards
        rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        return rewards
    
    def get_torch_data(self, gamma, N=10):
        all_states = []
        all_actions = []
        all_rewards = []
        all_action_idxs = []
        all_logprobs = []
        # all_dones = []
        for X in [self.all_reward_trajectories, self.all_no_reward_trajectories]:
            try:
                idxs = np.random.randint(0, len(X),size=N)
            except:
                idxs = [] # len(self.all_reward_traj) == 0
            for idx in idxs:
                traj = X[idx]
                rewards = []
                discounted_reward = 0
                for reward in reversed(traj.rewards):
                    discounted_reward = reward + (gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)
                all_rewards += rewards # extend list
                all_states += traj.states
                all_actions += traj.actions
                all_action_idxs += traj.act_idxs
                all_logprobs += traj.logprobs
                # all_dones += traj.dones
        
        all_states = torch.squeeze(torch.tensor(np.array(all_states))).detach().to(device).type(torch.float)
        all_action_idxs = torch.squeeze(torch.tensor(np.array(all_action_idxs))).detach().to(device).type(torch.int64)
        all_actions = torch.squeeze(torch.tensor(np.array(all_actions))).detach().to(device)
        all_logprobs = torch.squeeze(torch.tensor(all_logprobs)).detach().to(device)
        all_rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32).to(device)
        # all_dones = torch.tensor(np.array(all_dones), dtype=torch.float32).to(device)
        # all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-7)

        return all_states, all_actions, all_rewards, all_action_idxs, all_logprobs#, all_dones

    def get_states(self):
        all_states = []
        for traj in self.trajectories:
            all_states += traj.states
        return torch.squeeze(torch.tensor(all_states)).detach().to(device)

    def restart_traj(self):
        self.clear()

    def clear(self):
        self.all_reward_trajectories += [traj for traj in self.trajectories if traj.has_reward]
        self.all_no_reward_trajectories += [traj for traj in self.trajectories if not traj.has_reward]
        self.trajectories = []
        self.all_reward_trajectories = self.all_reward_trajectories[-self.max_:]
        self.all_no_reward_trajectories = self.all_no_reward_trajectories[-self.max_:]
        

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, param):
        super(ActorCritic, self).__init__()

        has_continuous_action_space = True
        self.has_continuous_action_space = has_continuous_action_space        
        
        if has_continuous_action_space:
            self.action_dim = action_dim['mdp'].shape[0]
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)

        self.actor = nn.Sequential(
                        nn.Linear(state_dim['mdp'].shape[0], 64),
                        nn.ReLU(), #relu for other experiments
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.action_dim)
                    )
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim['mdp'].shape[0], 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()

            clipped_action = torch.clip(action, -1, 1)
            action = clipped_action

            action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        if self.has_continuous_action_space:

            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action_logprobs = dist.log_prob(action)
            
            log_probs = action_logprobs
            
            # State values 
            state_values = self.critic(state)

            dist_gaussian = dist.entropy()
            dist_entropy = dist_gaussian
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(state)
        
        return log_probs, state_values, dist_entropy
