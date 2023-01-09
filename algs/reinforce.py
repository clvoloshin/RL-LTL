import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np
import logger
import time
from policies.policy import Policy
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
from torch.autograd import Variable
import math


pi = Variable(torch.FloatTensor([math.pi]))

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class REINFORCE():
    def __init__(self, policy, optimizer, param, second_order = False):
    
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = param['gamma']
        self.saved_log_probs = []
        self.saved_states = []
        self.rewards = []
        self.second_order = second_order

    # def select_action(self, state):
    #     state = torch.from_numpy(state).float().unsqueeze(0)
    #     probs = self.policy(state)
    #     m = Categorical(probs)
    #     action = m.sample()
    #     self.saved_log_probs.append(m.log_prob(action))
    #     return action.item()
    
    def select_action(self, state):
        (mu, sigma_sq), rho = self.policy(torch.tensor(np.atleast_2d(state), dtype=torch.float))
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps)).data
        prob = normal(action, mu, sigma_sq)
        #entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        self.saved_log_probs.append(log_prob)
        self.saved_states.append(state)
        return action#, log_prob, entropy

    def update(self, env):
        R = 0
        eps = np.finfo(np.float32).eps.item()
        policy_loss = []
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        _, rhos = self.policy(torch.tensor(self.saved_states, dtype=torch.float))
        signals = []
        for state in self.saved_states:
            _, signal = env.label(state)
            signals.append(signal['y'])
        with torch.no_grad():
            torch_signals = torch.tensor(signals)
        
        consistency_loss = (rhos[:-1] - torch.max(torch_signals, rhos[1:]))**2

        for log_prob, R, rho in zip(self.saved_log_probs, returns, rhos):
            rew = R #+ 2 * rho
            policy_loss.append(-log_prob * rew)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() #+ consistency_loss.sum()
        if self.second_order:
            self.optimizer.step(loss = policy_loss)
        else:
            policy_loss.backward()
            self.optimizer.step()

        
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.saved_states[:]

def run_reinforce(param, env, second_order = False):
    
    
    policy = Policy(env.observation_space, env.action_space['mdp'].shape[0])
    optimizer = torch.optim.Adam(policy.parameters(), lr=param['reinforce']['lr'])
    reinforce = REINFORCE(policy, optimizer, param, second_order)

    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in range(param['n_traj']):
        state, _ = env.reset()
        ep_reward = 0
        disc_ep_reward = 0
        

        for t in range(1, param['T']):  # Don't infinite loop while learning
            action = reinforce.select_action(state).squeeze(0).numpy()
            next_state, cost, done, info = env.step(action)
            reward = -cost

            # done = done or 'y' in info['label']

            # if i_episode % 20 == 0:
            #     env.render()
            reinforce.rewards.append(reward)
            ep_reward += reward
            disc_ep_reward += param['gamma']**(t-1) * reward

            if done:
                break
            state = next_state

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [disc_ep_reward]
            success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            method = 'TR' if second_order else 'Adam'
            plot_something_live(axes, [np.arange(len(history)),  np.arange(len(success_history))], [history, success_history], method)
            logger.logkv('Iteration', i_episode)
            logger.logkv('Method', method)
            logger.logkv('Success', success_history[-1])
            logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            logger.logkv('EpisodeReward', ep_reward)
            logger.logkv('DiscEpisodeReward', disc_ep_reward)
            logger.logkv('TimestepsAlive', avg_timesteps)
            logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['T'])

            _, rhos = reinforce.policy(torch.tensor(reinforce.saved_states[0:1], dtype=torch.float))
            logger.logkv('Rho[0]', rhos[0].detach().numpy())
            if second_order:
                logger.logkv('TrustRegion', optimizer.state['trust_region'])
                logger.logkv('AvgCGSteps', np.mean(optimizer.info['cg_steps']))
                logger.logkv('CGSteps', optimizer.info['cg_steps'][-1])
                logger.logkv('ExitCode', optimizer.info['code'])

            # if second_order: print(np.mean(optimizer.info['cg_steps']), optimizer.info['cg_steps'][-1])
            logger.dumpkvs()
        
        reinforce.update(env)
    
    plt.close()