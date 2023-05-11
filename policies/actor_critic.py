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
        self.buchis = []
        self.next_buchis = []
        self.act_idxs = []
        self.is_eps = []
        self.logprobs = []
        self.counter = 0
        self.done = False
        self.has_reward = False
        self.action_placeholder = action_placeholder # should be of MDP action shape
    
    def add(self, s, b, a, r, s_, b_, is_eps, act_idx, logprob):
        self.counter += 1
        # if r > 0: import pdb; pdb.set_trace()
        self.states.append(s)
        self.buchis.append(b)
        self.next_states.append(s_)
        self.next_buchis.append(b_)
        self.actions.append(a if not is_eps else self.action_placeholder)
        self.rewards.append(r)  # want this to hold the original MDP reward
        self.has_reward = self.has_reward or (r > 0)
        self.done = self.done or (r < 0)
        self.is_eps.append(is_eps)
        self.act_idxs.append(act_idx)
        self.logprobs.append(logprob)
    
    def get_last_buchi(self):
        return self.next_buchis[-1]
    
    def copy(self):
        # traj = Trajectory(self.action_placeholder)
        # traj.states = deepcopy(self.states)
        # traj.actions = deepcopy(self.actions)
        # traj.rewards = deepcopy(self.rewards)
        # traj.next_states = deepcopy(self.next_states)
        # traj.buchis = deepcopy(self.buchis)
        # traj.next_buchis = deepcopy(self.next_buchis)
        # traj.act_idxs = deepcopy(self.act_idxs)
        # traj.is_eps = deepcopy(self.is_eps)
        # traj.logprobs = deepcopy(self.logprobs)
        # traj.counter = deepcopy(self.counter)
        # return traj
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
    
    def add_experience(self, env, s, b, a, r, s_, b_, act_idx, is_eps, logprobs):
        
        if self.to_hallucinate:
            self.update_trajectories(env, s, b, a, r, s_, b_, act_idx, is_eps, logprobs)
            self.make_trajectories(env, s, b, a, r, s_, b_, act_idx, is_eps, logprobs)
        else:
            if len(self.trajectories) == 0: 
                traj = Trajectory(self.action_placeholder)
                self.trajectories.append(traj)
            
            traj = self.trajectories[-1]
            traj.add(s, b, a, r, s_, b_, is_eps, act_idx, logprobs[b][act_idx])

        
    def make_trajectories(self, env, s, b, a, r, s_, b_, act_idx, is_eps, logprobs):
        if not is_eps:
            assert act_idx == 0
            current_terminal_buchis = set([traj.get_last_buchi() for traj in self.trajectories if not traj.done])
            for buchi_state in range(env.observation_space['buchi'].n):
                if (self.main_trajectory is None) and (buchi_state == b): 
                    self.main_trajectory = len(self.trajectories)
                if buchi_state in current_terminal_buchis: continue
                # import pdb; pdb.set_trace()
                traj = Trajectory(self.action_placeholder)
                next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
                if accepting_rejecting_neutal < 1: 
                    traj.add(s, buchi_state, a, accepting_rejecting_neutal, s_, next_buchi_state, is_eps, act_idx, logprobs[buchi_state][act_idx])
                    self.trajectories.append(traj)
            
                # also add epsilon transition 
                try:                        
                    for eps_idx in range(env.action_space[buchi_state].n):
                        # import pdb; pdb.set_trace()
                        traj = Trajectory(self.action_placeholder)
                        
                        # make epsilon transition
                        next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s, buchi_state, eps_idx)
                        traj.add(s, buchi_state, a, accepting_rejecting_neutal, s, next_buchi_state, True, 1 + eps_idx, logprobs[buchi_state][1 + eps_idx])
                    
                        # resync trajectory with s_
                        next_next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, next_buchi_state)
                        traj.add(s, next_buchi_state, a, accepting_rejecting_neutal, s_, next_next_buchi_state, is_eps, act_idx, logprobs[next_buchi_state][act_idx])
                        self.trajectories.append(traj)
                except:
                    pass
        else:
            pass

    def update_trajectories(self, env, s, b, a, r, s_, b_, act_idx, is_eps, logprobs):
        new_trajectories = []
        if not is_eps:
            # update all trajectories
            for traj in self.trajectories:
                if traj.done == True: continue
                buchi_state = traj.get_last_buchi()
            
                # First add epsilon transitions if possible
                try:                        
                    for eps_idx in range(env.action_space[buchi_state].n):
                        # tic = time.time()
                        traj_copy = traj.copy()
                        # traj_copy = Trajectory(self.action_placeholder)
                        # print('CopyTime', time.time() - tic)
                        
                        # make epsilon transition
                        next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s, buchi_state, eps_idx)
                        traj_copy.add(s, buchi_state, a, accepting_rejecting_neutal, s, next_buchi_state, True, 1 + eps_idx, logprobs[buchi_state][1 + eps_idx])
                    
                        # resync trajectory with s_
                        next_next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, next_buchi_state)
                        traj_copy.add(s, next_buchi_state, a, accepting_rejecting_neutal, s_, next_next_buchi_state, is_eps, act_idx, logprobs[next_buchi_state][act_idx])
                        new_trajectories.append(traj_copy)
                except:
                    pass

                next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
                traj.add(s, buchi_state, a, accepting_rejecting_neutal, s_, next_buchi_state, is_eps, act_idx, logprobs[buchi_state][act_idx])
        else:
            # only update main, non-hallucinated, trajectory.
            if len(self.trajectories) == 0: return
            traj = self.trajectories[self.main_trajectory]
            buchi_state = traj.get_last_buchi()
            next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
            traj.add(s, buchi_state, a, accepting_rejecting_neutal, s_, next_buchi_state, is_eps, act_idx, logprobs[buchi_state][act_idx])
        
        for traj in new_trajectories:
            self.trajectories.append(traj)
    
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
        all_buchis = []
        all_actions = []
        all_rewards = []
        all_action_idxs = []
        all_logprobs = []
        all_next_buchis = []
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
                all_buchis += traj.buchis
                all_next_buchis += traj.next_buchis
                # all_dones += traj.dones
        
        all_states = torch.squeeze(torch.tensor(np.array(all_states))).detach().to(device).type(torch.float)
        all_buchis = torch.squeeze(torch.tensor(np.array(all_buchis))).detach().to(device).type(torch.int64).unsqueeze(1).unsqueeze(1)
        all_next_buchis = torch.squeeze(torch.tensor(np.array(all_next_buchis))).detach().to(device).type(torch.int64).unsqueeze(1).unsqueeze(1)
        all_action_idxs = torch.squeeze(torch.tensor(np.array(all_action_idxs))).detach().to(device).type(torch.int64)
        all_actions = torch.squeeze(torch.tensor(np.array(all_actions))).detach().to(device)
        all_logprobs = torch.squeeze(torch.tensor(all_logprobs)).detach().to(device)
        all_rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32).to(device)
        # all_dones = torch.tensor(np.array(all_dones), dtype=torch.float32).to(device)
        # all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-7)

        return all_states, all_buchis, all_actions, all_next_buchis, all_rewards, all_action_idxs, all_logprobs#, all_dones

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
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim['mdp'].shape[0], 64),
                            nn.ReLU(), #relu for other experiments
                            nn.Linear(64, 64),
                            nn.ReLU(),
                        )
            
            self.main_head = nn.Sequential(
                            nn.Linear(64, state_dim['buchi'].n * self.action_dim),
                            nn.Tanh()
                        )
            self.action_switch = nn.Linear(64, state_dim['buchi'].n * action_dim['total']) # for buchi epsilons

            with torch.no_grad():
                # bias towards no epsilons in the beginning
                self.action_switch.bias[::action_dim['total']] = 5.

                # bias towards left turn
                # self.main_head[0].bias[::self.action_dim] = 0
                                        

            self.main_shp = (state_dim['buchi'].n, self.action_dim)
            self.shp = (state_dim['buchi'].n, action_dim['total'])
            
            # Mask actions not available
            self.mask = torch.ones((state_dim['buchi'].n, action_dim['total'])).type(torch.bool)
            if action_dim['total'] > 1:
                for buchi in range(state_dim['buchi'].n):
                    try:
                        eps = action_dim['total'] - 1 + action_dim[buchi].n
                    except:
                        eps = action_dim['total'] - 1
                    self.mask[buchi, eps:] = False                        
        else:
            raise NotImplemented
            # self.actor = nn.Sequential(
            #                 nn.Linear(state_dim, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, 64),
            #                 nn.Tanh(),
            #                 nn.Linear(64, action_dim),
            #                 nn.Softmax(dim=-1)
            #             )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim['mdp'].shape[0], 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, state_dim['buchi'].n)
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
    
    def act(self, state, buchi_state):
        
        is_eps = False

        if self.has_continuous_action_space:
            body = self.actor(state)
            action_head = self.main_head(body)
            action_mean = torch.reshape(action_head, self.main_shp)[buchi_state]

            # action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
            # masked_head_all = torch.masked.MaskedTensor(action_switch_head_all, self.mask)
            # probs_all = F.softmax(masked_head_all)
            # probs = probs_all.to_tensor(0)
            
            action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
            action_switch = action_switch_head_all#[buchi_state]
            mask = self.mask#[buchi_state]

            ### Bias against epsilon actions by multiplication by +10
            # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
            
            # Fix
            probs = self.masked_softmax(action_switch, mask, -1)
            # probs = probs_all.squeeze()

            # action_switch_head = action_switch_head_all[buchi_state]
            # masked_head = action_switch_head[:self.mask[buchi_state].sum()]
            # probs = F.softmax(masked_head)
            try:
                action_or_eps = Categorical(probs[buchi_state])
            except:
                import pdb; pdb.set_trace()
            act_or_eps = action_or_eps.sample()

            if act_or_eps == 0:
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
                dist = MultivariateNormal(action_mean, cov_mat)
                action = dist.sample()

                clipped_action = torch.clip(action, -1, 1)
                action = clipped_action

                action_logprob = dist.log_prob(action) + action_or_eps.log_prob(act_or_eps)
            else:
                is_eps = True
                action = act_or_eps
                action_logprob = action_or_eps.log_prob(act_or_eps)
            
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.detach(), int(act_or_eps.detach()), is_eps, action_logprob.detach(), torch.log(probs).detach()
    
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

    def evaluate(self, state, buchi, action, action_idxs):
        if self.has_continuous_action_space:

            body = self.actor(state)
            action_head = self.main_head(body)
            action_means = torch.reshape(action_head, (-1,) + self.main_shp)
            action_mean = torch.take_along_dim(action_means, buchi, dim=1).squeeze()

            action_switch_head_all = torch.reshape(self.action_switch(body), (-1,) + self.shp)
            action_switch = torch.take_along_dim(action_switch_head_all, buchi, dim=1)
            mask = torch.take_along_dim(self.mask.unsqueeze(0).repeat(len(action), 1, 1).to(device), buchi, dim=1)

            ### Bias against epsilon actions by multiplication by +10
            # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
            
            ### Bug: Disturbs gradient flow
            # masked_switch = torch.masked.MaskedTensor(action_switch, mask)
            # probs_all = F.softmax(masked_switch, dim=-1)

            # Fix
            probs_all = self.masked_softmax(action_switch, mask, -1)
            probs = probs_all.squeeze()
            dist_coinflip = Categorical(probs)
            

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            
            action_logprobs = dist.log_prob(action)
            try:
                logprobs_from_coinflip = torch.log(torch.take_along_dim(probs, action_idxs.unsqueeze(1), dim=1).squeeze() + 1e-8)
            except:
                logprobs_from_coinflip = torch.log(probs)

            # If controller take (a == 0) then LOG(P(a==0) * Normal(A))
            # If controller take (epsilon transition) then LOG(P(A))
            log_probs = logprobs_from_coinflip + action_logprobs * (action_idxs == 0)
            
            # State values 
            state_values = torch.take_along_dim(self.critic(state), buchi.squeeze(-1), dim=1).squeeze()

            # Entropy. Overapprox, not exact. RECHECK
            dist_coinflip = dist_coinflip.entropy()
            dist_gaussian = dist.entropy()
            dist_entropy = dist_coinflip + dist_gaussian * (action_idxs == 0)
            
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
