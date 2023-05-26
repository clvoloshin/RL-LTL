import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np

from policies.actor_critic import ActorCritic, RolloutBuffer
from stlrl import Robustness

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
    def __init__(self, state_dim, action_dim, gamma, param):

        lr_rho = param['robustness']['lr_rho']
        lr_actor = param['ppo']['lr_actor']
        lr_critic = param['ppo']['lr_critic']
        K_epochs = param['ppo']['K_epochs']
        eps_clip = param['ppo']['eps_clip']
        has_continuous_action_space = True
        action_std_init = param['ppo']['action_std']

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.rho = Robustness(state_dim, param['ltl']['formula'])

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                        
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.rho.copy_new_to_old()
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std * action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        with torch.no_grad():
            mdp_state = torch.FloatTensor(state['mdp']).to(device)
            buchi_state = state['buchi']
            action, action_logprob, is_eps = self.policy_old.act(mdp_state, buchi_state)

        self.buffer.states.append(mdp_state)
        self.buffer.buchi_states.append(buchi_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.is_eps.append(is_eps)

        if self.has_continuous_action_space:      
            return action.detach().cpu().numpy().flatten(), is_eps
        else:
            return action.item(), is_eps
        
    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # get robustness values
            # scores, rho_losses = self.rho.score(old_states, self.buffer.atomics)
            # # print(scores)
            # with torch.no_grad():
            #     scores = scores[0:1, 0]

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()

            # MODIFY ADVANTAGE *****
            # advantages = advantages[:-1] + 
            # advantages = scores

            surr1 = ratios[:-1] * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_grad = -torch.min(surr1, surr2) 
            val_loss = 0.5*self.MseLoss(state_values, rewards) 
            entropy_loss = - 0.01*dist_entropy
            loss = policy_grad + val_loss + entropy_loss 
            # logger.logkv('score', scores[0])
            logger.logkv('policy_grad', policy_grad.detach().mean())
            logger.logkv('val_loss', val_loss.detach().mean())
            logger.logkv('entropy_loss', entropy_loss.detach().mean())
            # logger.logkv('rho_losses', rho_losses.detach().mean())

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.rho.copy_new_to_old()

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def run_ppo_guided(param, env, second_order = False):
    
    agent = PPO(env.observation_space, env.action_space, param['gamma'], param)
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
            action, is_eps = agent.select_action(state)
            # action = reinforce.select_action(state).squeeze(0).numpy()
            next_state, cost, done, info = env.step(action, is_eps)
            reward = -cost

            # done = done or 'y' in info['label']

            if i_episode % 20 == 0:
                env.render()
            # agent.buffer.atomics.append(info['signal'])
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            ep_reward += reward
            disc_ep_reward += param['gamma']**(t-1) * reward

            if done:
                break
            state = next_state

        import pdb; pdb.set_trace()
        for _ in range(t):
            agent.buffer.q_desired.append(q_desired)
            agent.buffer.q_realized.append(env.automaton.get_state())
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
            logger.logkv('ActionSTD', agent.action_std)

            # _, rhos = reinforce.policy(torch.tensor(reinforce.saved_states[0:1], dtype=torch.float))
            # logger.logkv('Rho[0]', rhos[0].detach().numpy())
            
            # if second_order: print(np.mean(optimizer.info['cg_steps']), optimizer.info['cg_steps'][-1])
            logger.dumpkvs()
        
        if i_episode % param['ppo']['update_timestep'] == 0:
            agent.update()
        
        if i_episode % param['ppo']['action_std_decay_freq'] == 0:
            agent.decay_action_std(param['ppo']['action_std_decay_rate'], param['ppo']['min_action_std'])
    
    plt.close()