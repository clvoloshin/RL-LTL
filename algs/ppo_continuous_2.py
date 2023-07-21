import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from utls.utls import parse_stl_into_tree

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
    def __init__(self, 
            env_space, 
            act_space, 
            gamma, 
            param, 
            to_hallucinate=False
            ) -> None:

        lr_actor = param['ppo']['lr_actor']
        lr_critic = param['ppo']['lr_critic']
        self.K_epochs = param['ppo']['K_epochs']
        self.batch_size = param['ppo']['batch_size']
        self.eps_clip = param['ppo']['eps_clip']
        self.has_continuous_action_space = True
        action_std_init = param['ppo']['action_std']
        self.temp = param['ppo']['action_std']
        self.alpha = param['ppo']['alpha']
        self.ltl_lambda = param['lambda']

        self.policy = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.mean_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.log_std_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.action_switch.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer(
            env_space['mdp'].shape, 
            act_space['mdp'].shape, 
            param['replay_buffer_size'], 
            to_hallucinate)
        
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
            action, action_mean, action_idx, is_eps, action_logprob, all_logprobs = self.policy_old.act(state_tensor, buchi)
            if is_testing:
                return action_mean, action_idx, is_eps, action_logprob, all_logprobs
            return action, action_idx, is_eps, action_logprob, all_logprobs
        
    def collect(self, s, b, a, r, s_, b_):
        # TODO: update this part
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
        #print(f'Setting temperature: {self.temp}')
        self.set_action_std(self.temp)

    def update(self):
        self.num_updates_called += 1

        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            
            
            # Get data from random reward-ful trajectories
            # old_states, old_buchis, old_actions, old_next_buchis, rewards, \
            #     old_action_idxs, old_logprobs = self.buffer.get_torch_data(
            #         self.gamma, self.batch_size)
            
            # TODO: update the RolloutBuffer to return rhos, edge, terminal
            old_states, old_buchis, old_actions, old_next_buchis, rewards, \
                lrewards, crewards, old_action_idxs, old_logprobs, old_rhos, old_edges, old_terminals \
                    = self.buffer.get_torch_data(
                        self.gamma, self.batch_size
                    )
            
            if len(old_states) == 0:
                # No signal available
                self.buffer.clear()
                return
            # use constrained optimization to modify the reward based on current lambda
            #TODO: support other LTL reward types instead of just one
            #TODO: fix return type and function signature of this?
            # rewards, _ = self.reward_funcs(None, None, old_buchis, old_next_buchis, rewards)
            
            # calculate the constrained reward
            # rewards, _, info = self.constrained_reward(rhos, edge, terminal, old_buchis,\
            #                                   old_next_buchis, rewards)
            # if self.num_updates_called >= 2950 and self.num_updates_called % 25 == 0:
            #     import pdb; pdb.set_trace()
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_buchis, old_actions, old_action_idxs)
            
            
            # take the cycle rewards, and find the cycle that maximizes the summed reward
            best_cycle_idx = torch.argmax(crewards.sum(dim=0)).item()
            crewards = crewards[:, best_cycle_idx]
            # if best_cycle_idx == 2:
            #     import pdb; pdb.set_trace()

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = crewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_grad = -torch.min(surr1, surr2) 
            val_loss = self.MseLoss(state_values, crewards) 
            entropy_loss = dist_entropy
            # if (rewards == 0).all():
            #     # No signal available
            #     loss = 0.5*val_loss #- 0.01*entropy_loss 
            # else:
            normalized_val_loss = val_loss / (self.ltl_lambda / (1 - self.gamma))

            loss = policy_grad + 0.5*normalized_val_loss - self.alpha*entropy_loss #TODO: tune the amount we want to regularize entropy
            logger.logkv('policy_grad', policy_grad.detach().mean())
            logger.logkv('val_loss', val_loss.detach().mean())
            logger.logkv('entropy_loss', entropy_loss.detach().mean())
            logger.logkv('rewards', crewards.mean())
            # take gradient step

            self.optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        # if self.num_updates_called > 125 and self.num_updates_called % 25 == 0:
        #     import pdb; pdb.set_trace()
        self.buffer.clear()
        return loss.mean(), {"policy_grad": policy_grad.detach().mean(), "val_loss": normalized_val_loss.detach().item(), "entropy_loss": entropy_loss.detach().mean()}
    

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    mdp_ep_reward = 0
    ltl_ep_reward = 0
    disc_ep_reward = 0
    if not testing: 
        agent.buffer.restart_traj()
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
        
        # TODO: update the env_step function to return rhos, edge, terminal as info
        try:
            next_state, mdp_reward, done, info = env.step(action, is_eps)
        except:
            next_state, mdp_reward, done, _, info = env.step(action, is_eps)
        #reward = int(info['is_accepting'])
        rhos = info['rho']
        edge = info['edge']
        terminal = info['is_rejecting']
        constrained_reward, _, rew_info = env.constrained_reward(rhos, terminal, state['buchi'], next_state['buchi'], mdp_reward)
        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
            #print(next_state['mdp'])
            #print(next_state['buchi'])
            # try:
            #     print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
            # except:
            #     pass
            # print(action)
            # print(agent.Q(s, b))
        # tic = time.time()
        agent.buffer.add_experience(
            env, 
            state['mdp'], 
            state['buchi'], 
            action, 
            mdp_reward,
            rew_info["ltl_reward"],
            constrained_reward, 
            next_state['mdp'], 
            next_state['buchi'], 
            action_idx, 
            is_eps, 
            all_logprobs,
            rhos,
            edge,
            terminal,
            )
        # total_experience_time += time.time() - tic

        # if visualize:
        #     env.render()
        # agent.buffer.atomics.append(info['signal'])
        mdp_ep_reward += rew_info["mdp_reward"]
        ltl_ep_reward += rew_info["ltl_reward"]
        disc_ep_reward += param['gamma']**(t-1) * mdp_reward # TODO: change this to represent combined reward

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
    #print(next_state['mdp'])
    #print(next_state['buchi'])
    # try:
    #     print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
    # except:
    #     pass
    #print(action)
    return mdp_ep_reward, ltl_ep_reward, t
        
def run_ppo_continuous_2(param, runner, env, second_order = False, to_hallucinate=False, visualize=True):
    
    ## G(F(g) & ~b & ~r & ~y)
    #constrained_rew_fxn = {0: [env.automaton.edges(0, 1)[0], env.automaton.edges(0, 0)[0]], 1: [env.automaton.edges(1, 0)[0]]}
    ## G(F(y & X(F(r)))) & G~b
    constrained_rew_fxn = {0: [env.automaton.edges(0, 1)[0]], 1: [env.automaton.edges(1, 2)[0]], 2: [env.automaton.edges(2, 0)[0]]}
    #import pdb; pdb.set_trace()
    ## F(G(y))
    #constrained_rew_fxn = {1: [env.automaton.edges(1, 1)[0]]}
    
    ## F(r & XF(G(y)))
    #constrained_rew_fxn = {2: [env.automaton.edges(2, 2)[0]]}  
    #import pdb; pdb.set_trace()
    ## F(r & XF(g & XF(y))) & G~b
    # constrained_rew_fxn = {2: [env.automaton.edges(2, 3)[0]], 3: [env.automaton.edges(3, 1)[0]], 1: [env.automaton.edges(1, 0)[0]], 0: [env.automaton.edges(0, 0)[0]]}  
    # constrained_rew_fxn = {0: [env.automaton.edges(0, 0)[0]]}

    stl_tree = parse_stl_into_tree(param['ltl']['formula'])
    agent = PPO(
        env.observation_space, 
        env.action_space, 
        param['gamma'], 
        param, 
        to_hallucinate)
    
    # fig, axes = plt.subplots(2, 1)
    # history = []
    # success_history = []
    # disc_success_history = []
    fixed_state, _ = env.reset()
    #runner.log({"testing": wandb.Image(env.render(states=[fixed_state['mdp']], save_dir=None))})


    for i_episode in tqdm(range(param['ppo']['n_traj'])):
        # TRAINING

        # Get trajectory
        # tic = time.time()
        mdp_ep_reward, ltl_ep_reward, t = rollout(env, agent, param, i_episode, runner, testing=False)
        # toc = time.time() - tic
        # print('Rollout Time', toc)

        # update weights
        # tic = time.time()
        losstuple = agent.update()
        if losstuple is not None:
            current_loss, loss_info = losstuple
        # toc2 = time.time() - tic
        # print(toc, toc2)
        
        if i_episode % param['ppo']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['ppo']['temp_decay_rate'], param['ppo']['min_action_temp'], param['ppo']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                mdp_test_reward, ltl_test_reward, t = rollout(env, agent, param, i_episode, runner, testing=True, visualize=visualize ) #param['n_traj']-100) ))
            test_data = np.array(test_data)
    
        if i_episode > 1 and i_episode % 1 == 0:
            runner.log({#'Iteration': i_episode,
                    #  'last_reward': last_reward,
                    #  'Method': method,
                    #  'Success': success_history[-1],
                    #  'Last20Success': np.mean(np.array(success_history[-20:])),
                    #  'DiscSuccess': disc_success_history[-1],
                    #  'Last20DiscSuccess': np.mean(np.array(disc_success_history[-20:])),
                    #  'EpisodeReward': ep_reward,
                    #  'DiscEpisodeReward': disc_ep_reward,
                    # 'EpisodeRhos': stl_val,
                    # 'ExpectedQVal': max_q_val,
                    'R_LTL': ltl_ep_reward,
                    'R_MDP': mdp_ep_reward,
                    'LossVal': current_loss,
                    #'AvgTimesteps': t,
                    "PolicyGradLoss": loss_info["policy_grad"],
                    "MSEValLoss": loss_info["val_loss"],
                    #  'TimestepsAlive': avg_timesteps,
                    #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                     'ActionTemp': agent.temp,
                     'EntropyLoss': loss_info["entropy_loss"],
                     "Test_R_LTL": ltl_test_reward,
                     "Test_R_MDP": mdp_test_reward
                     })
            
            # avg_timesteps = t #np.mean(timesteps)
            #history += [disc_ep_reward]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            # success_history += [test_data[:, 0].mean()]
            # disc_success_history += [test_data[:, 1].mean()]
            method = "PPO"
            #plot_something_live(axes, [np.arange(len(history)),  np.arange(len(success_history))], [history, success_history], method)
            # logger.logkv('Iteration', i_episode)
            # logger.logkv('Method', method)
            # logger.logkv('Success', success_history[-1])
            # logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            # logger.logkv('DiscSuccess', disc_success_history[-1])
            # logger.logkv('Last20DiscSuccess', np.mean(np.array(disc_success_history[-20:])))
            # logger.logkv('EpisodeReward', ep_reward)
            # logger.logkv('DiscEpisodeReward', disc_ep_reward)
            # logger.logkv('TimestepsAlive', avg_timesteps)
            # logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['ppo']['T'])
            # logger.logkv('ActionTemp', agent.temp)
            # logger.dumpkvs()
            
    plt.close()