import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from policies.actor_critic import RolloutBuffer, ActorCritic
import time

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


class PPO:
    def __init__(self, env_space, act_space, gamma, param) -> None:

        lr_actor = param['ppo']['lr_actor']
        lr_critic = param['ppo']['lr_critic']
        self.K_epochs = param['ppo']['K_epochs']
        self.eps_clip = param['ppo']['eps_clip']
        self.has_continuous_action_space = True
        action_std_init = param['ppo']['action_std']
        self.temp = param['ppo']['action_std']

        self.policy = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.main_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.action_switch.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer(env_space['mdp'].shape, act_space['mdp'].shape, param['replay_buffer_size'])
        
        self.gamma = gamma
        self.num_updates_called = 0
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state, is_testing):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state['mdp']).to(device)
            buchi = state['buchi']
            action, action_idx, is_eps, action_logprob, all_logprobs = self.policy_old.act(state_tensor, buchi)

            return action, action_idx, is_eps, action_logprob, all_logprobs
        
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
        self.set_action_std(self.temp)
    
    def update(self):
        self.num_updates_called += 1

        # Optimize policy for K epochs
        for k in range(self.K_epochs):

            # Get data from random reward-ful trajectories
            old_states, old_buchis, old_actions, rewards, old_action_idxs, old_logprobs = self.buffer.get_torch_data(self.gamma)
            if len(old_states) == 0:
                # No signal available
                self.buffer.clear()
                return

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_buchis, old_actions, old_action_idxs)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_grad = -torch.min(surr1, surr2) 
            val_loss = self.MseLoss(state_values, rewards) 
            entropy_loss = dist_entropy
            
            # if (rewards == 0).all():
            #     # No signal available
            #     loss = 0.5*val_loss #- 0.01*entropy_loss 
            # else:
            loss = policy_grad + 0.5*val_loss #- 0.01*entropy_loss 
            logger.logkv('policy_grad', policy_grad.detach().mean())
            logger.logkv('val_loss', val_loss.detach().mean())
            logger.logkv('entropy_loss', entropy_loss.detach().mean())
            logger.logkv('rewards', rewards.mean())
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    

def rollout(env, agent, param, i_episode, testing=False, visualize=False):
    state, _ = env.reset()
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    if testing & visualize:
        s = torch.tensor(state['mdp']).type(torch.float)
        b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
        print(0, state['mdp'], state['buchi'])
        # print(agent.Q(s, b))
    
    for t in range(1, param['T']):  # Don't infinite loop while learning
        # tic = time.time()
        action, action_idx, is_eps, log_prob, all_logprobs = agent.select_action(state, testing)
        # print('Get Action', time.time() - tic)

        if is_eps:
            action = int(action)
        else:
            action = action.numpy().flatten()
        
        next_state, cost, done, info = env.step(action, is_eps)
        reward = int(info['is_accepting'])
        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
            print(t, next_state['mdp'], next_state['buchi'], action)
            # print(agent.Q(s, b))

        # tic = time.time()
        agent.buffer.add_experience(env, state['mdp'], state['buchi'], action, info['is_accepting'], next_state['mdp'], next_state['buchi'], action_idx, is_eps, all_logprobs)
        # print('Get Experience', time.time() - tic)

        if visualize:
            env.render()
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        if done:
            break
        state = next_state

    return ep_reward, disc_ep_reward, t
        
def run_ppo_continuous(param, env, second_order = False):
    
    agent = PPO(env.observation_space, env.action_space, param['gamma'], param)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in range(param['n_traj']):
        # TRAINING

        # Get trajectory
        # tic = time.time()
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, testing=False)
        # toc = time.time() - tic

        # update weights
        # tic = time.time()
        agent.update()
        # toc2 = time.time() - tic
        
        if i_episode % param['ppo']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['ppo']['temp_decay_rate'], param['ppo']['min_action_temp'], param['ppo']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, test_iter, testing=True, visualize= ((i_episode % 10) == 0) & (test_iter == 0) ))
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
            # logger.logkv('DataCollectTime', toc)
            # logger.logkv('UpdateTime', toc2)
            
            logger.dumpkvs()
            
    plt.close()