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
import wandb
from tqdm import tqdm

################################## set device ##################################
# print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
# torch.default_device(device)

class PPO:
    def __init__(self, env_space, act_space, gamma, param, to_hallucinate=False) -> None:

        lr_actor = param['ppo']['lr_actor']
        lr_critic = param['ppo']['lr_critic']
        self.K_epochs = param['ppo']['K_epochs']
        self.batch_size = param['ppo']['batch_size']
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

        self.buffer = RolloutBuffer(env_space['mdp'].shape, act_space['mdp'].shape, param['replay_buffer_size'], to_hallucinate)
        
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
    
    def update(self, constrained_rew_fxn):
        self.num_updates_called += 1

        # Optimize policy for K epochs
        for k in range(self.K_epochs):

            # Get data from random reward-ful trajectories
            old_states, old_buchis, old_actions, old_next_buchis, rewards, old_action_idxs, old_logprobs = self.buffer.get_torch_data(self.gamma, self.batch_size)
            if len(old_states) == 0:
                # No signal available
                self.buffer.clear()
                return
            # use constrained optimization to modify the reward based on current lambda
            #TODO: support other LTL reward types instead of just one
            #TODO: fix return type and function signature of this?
            rewards, _ = constrained_rew_fxn(None, None, old_buchis, old_next_buchis, rewards)
            
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
    

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    # if testing & visualize:
    #     s = torch.tensor(state['mdp']).type(torch.float)
    #     b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
    #     print(0, state['mdp'], state['buchi'])
    #     # print(agent.Q(s, b))
    
    # total_action_time = 0
    # total_experience_time = 0
    
    for t in range(1, param['ppo']['T']):  # Don't infinite loop while learning
        # tic = time.time()
        action, action_idx, is_eps, log_prob, all_logprobs = agent.select_action(state, testing)
        # total_action_time += time.time() - tic 

        if is_eps:
            action = int(action)
        else:
            action = action.cpu().numpy().flatten()
        
        try:
            next_state, cost, done, info = env.step(action, is_eps)
        except:
            next_state, cost, done, _, info = env.step(action, is_eps)
        reward = int(info['is_accepting'])
        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
            print(next_state['mdp'])
            print(next_state['buchi'])
            try:
                print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
            except:
                pass
            print(action)
            # print(agent.Q(s, b))

        # tic = time.time()
        agent.buffer.add_experience(env, state['mdp'], state['buchi'], action, info['is_accepting'], next_state['mdp'], next_state['buchi'], action_idx, is_eps, all_logprobs)
        # total_experience_time += time.time() - tic

        # if visualize:
        #     env.render()
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        states.append(next_state['mdp'])
        buchis.append(next_state['buchi'])
        if done:
            break
        state = next_state

    if visualize:
        # frames = 
        # runner.log({"video": wandb.Video([env.render(states=np.atleast_2d(state), save_dir=None) for state in states], fps=10)})
        if testing: 
            runner.log({"testing": wandb.Image(env.render(states=states, save_dir=None))})
        else:
            runner.log({"training": wandb.Image(env.render(states=states, save_dir=None))})
    # print('Get Experience', total_experience_time)
    # print('Get Action', total_action_time)
    print(next_state['mdp'])
    print(next_state['buchi'])
    try:
        print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
    except:
        pass
    print(action)
    return ep_reward, disc_ep_reward, t
        
def run_ppo_continuous(param, env, constrained_rew_fxn, runner, second_order = False, to_hallucinate=False):
    
    agent = PPO(env.observation_space, env.action_space, param['gamma'], param, to_hallucinate)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    disc_success_history = []
    fixed_state, _ = env.reset()

    for i_episode in tqdm(range(param['ppo']['n_traj'])):
        # TRAINING

        # Get trajectory
        # tic = time.time()
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, runner, testing=False)
        # toc = time.time() - tic
        # print('Rollout Time', toc)

        # update weights
        # tic = time.time()
        agent.update(constrained_rew_fxn)
        # toc2 = time.time() - tic
        # print(toc, toc2)
        
        if i_episode % param['ppo']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['ppo']['temp_decay_rate'], param['ppo']['min_action_temp'], param['ppo']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, i_episode, testing=True, visualize= test_iter == 0 )) #param['n_traj']-100) ))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [disc_ep_reward]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            success_history += [test_data[:, 0].mean()]
            disc_success_history += [test_data[:, 1].mean()]
            method = "PPO"
            plot_something_live(axes, [np.arange(len(history)),  np.arange(len(success_history))], [history, success_history], method)
            logger.logkv('Iteration', i_episode)
            logger.logkv('Method', method)
            logger.logkv('Success', success_history[-1])
            logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            logger.logkv('DiscSuccess', disc_success_history[-1])
            logger.logkv('Last20DiscSuccess', np.mean(np.array(disc_success_history[-20:])))
            logger.logkv('EpisodeReward', ep_reward)
            logger.logkv('DiscEpisodeReward', disc_ep_reward)
            logger.logkv('TimestepsAlive', avg_timesteps)
            logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['ppo']['T'])
            logger.logkv('ActionTemp', agent.temp)
            logger.dumpkvs()
            
    plt.close()