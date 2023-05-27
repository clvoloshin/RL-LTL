import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import logger
from utls.plotutils import plot_something_live
import numpy as np
import numpy.ma as ma
import pandas as pd

class Buffer:
    def __init__(self, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1

        self.states = np.array([0 for _ in range(max_)])
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.next_states = np.array([0 for _ in range(max_)])
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

class Q_learning:
    def __init__(self, env_space, act_space, gamma, param) -> None:
        self.buffer = Buffer(param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        
        self.Q = ma.zeros((env_space['mdp'].n, env_space['buchi'].n, act_space['total']))
        self.N = ma.zeros((env_space['mdp'].n, env_space['buchi'].n, act_space['total']))
        
        # Mask actions not available
        for buchi in range(env_space['buchi'].n):
            try:
                eps = act_space['total'] - 1 + act_space[buchi].n
            except:
                eps = act_space['total'] - 1
            self.Q[:, buchi, eps:] = ma.masked
            self.N[:, buchi, eps:] = ma.masked
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
    
    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp): #argmax, breaking ties by picking least visited
            qs = self.Q[state['mdp'], state['buchi']]
            act = np.random.choice(np.where(qs == qs.max())[0])
        else: # uniformly random
            X = self.Q[state['mdp'], state['buchi']]
            pos = np.random.choice(X.count(), size=1)
            idx = np.take((~X.mask).nonzero(), pos, axis=1)
            act = idx[0][0]
        
        self.N[state['mdp'], state['buchi'], act] += 1
        is_eps = act >= self.n_mdp_actions
        return act, is_eps, 0
    
    def collect(self, s, b, a, r, s_, b_):
        self.buffer.add(s, b, a, r, s_, b_)
    
    def decay_temp(self, decay_rate, min_temp, decay_type):
        
        if decay_type == 'linear':
            self.temp = self.temp - decay_rate
        elif decay_type == 'exponential':
            self.temp = self.temp * decay_rate
        else:
            raise NotImplemented
        
        if (self.temp <= min_temp):
            self.temp = min_temp
        
        self.temp = round(self.temp, 4)
        print(f'Setting temperature: {self.temp}')
    
    def update(self):
        # for _ in range(self.n_batches):
        #     s, b, a, r, s_, b_ = self.buffer.get_all()
        #     self.Q[s.astype(int), b.astype(int), a.astype(int)] = r + self.gamma * self.Q[s_.astype(int), b_.astype(int)].max(axis=1)
        for _ in range(self.n_batches):
            s, b, a, r, s_, b_ = self.buffer.sample(self.batch_size)
            self.Q[s.astype(int), b.astype(int), a.astype(int)] = r + self.gamma * self.Q[s_.astype(int), b_.astype(int)].max(axis=1)
        return 0
        
def run_Q_discrete(param, env, second_order = False):
    
    agent = Q_learning(env.observation_space, env.action_space, param['gamma'], param)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in range(param['n_traj']):
        state, _ = env.reset()
        ep_reward = 0
        disc_ep_reward = 0
        agent.buffer.restart_traj()
        
        for t in range(1, param['T']):  # Don't infinite loop while learning
            action, is_eps, log_prob = agent.select_action(state)
            
            next_state, cost, done, info = env.step(action, is_eps)

            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, is_accepting = env.next_buchi(next_state['mdp'], buchi_state)
                    agent.collect(state['mdp'], buchi_state, action, is_accepting, next_state['mdp'], next_buchi_state)
                    if buchi_state == state['buchi']:
                        reward = is_accepting
                        agent.buffer.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            next_buchi_state, is_accepting = env.next_buchi(state['mdp'], buchi_state, eps_idx)
                            agent.collect(state['mdp'], buchi_state, env.action_space['mdp'].n + eps_idx, is_accepting, state['mdp'], next_buchi_state)    
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], state['buchi'], action, 0, next_state['mdp'], next_state['buchi'])
                agent.buffer.mark()
                reward = 0

            if i_episode % 50 == 0:
                env.render()
            # agent.buffer.atomics.append(info['signal'])
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
            logger.logkv('ActionTemp', agent.temp)
            
            logger.dumpkvs()
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'])
    
        
