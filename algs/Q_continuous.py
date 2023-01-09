import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from policies.dqn import Buffer, DQN

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


class Q_learning:
    def __init__(self, env_space, act_space, gamma, param) -> None:

        self.Q = DQN(env_space, act_space, param).to(device)
        self.Q_target = DQN(env_space, act_space, param).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.Q.actor.parameters(), 'lr': param['q_learning']['lr']},
                    ])

        self.update_target_network()

        self.buffer = Buffer(env_space['mdp'].shape, param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        self.iterations_per_target_update = param['q_learning']['iterations_per_target_update']
        self.iterations_since_last_target_update = 0
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
    
    def update_target_network(self):
        # copy current_network to target network
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                buchi = state['buchi']
                action, is_eps, action_logprob = self.Q.act(state_tensor, buchi)

            return action, is_eps, action_logprob
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                buchi = state['buchi']
                action, is_eps, action_logprob = self.Q.random_act(state, buchi)
            return action, is_eps, action_logprob

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
        for _ in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                s, b, a, r, s_, b_ = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float)
                b = torch.tensor(b).type(torch.int64).unsqueeze(1).unsqueeze(1)
                s_ = torch.tensor(s_).type(torch.float)
                b_ = torch.tensor(b_).type(torch.int64).unsqueeze(1).unsqueeze(1)
                r = torch.tensor(r)
                a = torch.tensor(a)
                targets = r + self.gamma * self.Q_target(s_, b_).amax(1)

            q_values = self.Q(s, b, False).gather(1, a.unsqueeze(1))
            # td_error = q_values - targets.to_tensor(0).unsqueeze(1).clone().detach()

            loss_func = torch.nn.SmoothL1Loss()
            loss = loss_func(q_values, targets.to_tensor(0).unsqueeze(1).clone().detach())

            # loss = (td_error**2).mean() # MSE

            # backward optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
                self.update_target_network()
                self.iterations_since_last_target_update = 0

def rollout(env, agent, param, i_episode, testing=False, visualize=False):
    state, _ = env.reset()
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    if testing & visualize:
        s = torch.tensor(state['mdp']).type(torch.float)
        b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
        print(0, state['mdp'], state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
        print(agent.Q(s, b))
    
    for t in range(1, param['T']):  # Don't infinite loop while learning
        action, is_eps, log_prob = agent.select_action(state, testing)
        
        next_state, cost, done, info = env.step(action, is_eps)
        reward = info['is_accepting']

        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
            print(t, next_state['mdp'], next_state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
            print(agent.Q(s, b))

        if not testing: # TRAIN ONLY
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

        if visualize:
            env.render()
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        if done:
            break
        state = next_state

    return ep_reward, disc_ep_reward, t
        
def run_Q_continuous(param, env, second_order = False):
    
    agent = Q_learning(env.observation_space, env.action_space, param['gamma'], param)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in range(param['n_traj']):
        # TRAINING
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, testing=False)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, test_iter, testing=True, visualize= ((i_episode % 50) == 0) & (test_iter == 0) ))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [disc_ep_reward]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            success_history += [test_data[:, 0].mean()]
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
            
    plt.close()