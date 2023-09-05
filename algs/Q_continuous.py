import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tqdm import tqdm
import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
import wandb
from policies.dqn import Buffer, DQN

# ################################## set device ##################################
# print("============================================================================================")
# # set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
# print("============================================================================================")


class Q_learning:
    def __init__(self, env_space, act_space, gamma, num_cycles, param) -> None:
        if len(env_space['mdp'].shape) == 0:
            envsize = (1,)
        else:
            envsize = env_space['mdp'].shape
        self.Q = DQN(env_space, act_space, param).to(device)
        self.Q_target = DQN(env_space, act_space, param).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.Q.actor.parameters(), 'lr': param['q_learning']['lr']},
                    ])

        self.update_target_network()
        self.num_cycles = num_cycles
        self.buffer = Buffer(envsize, self.num_cycles, param['replay_buffer_size'])
        self.good_buffer = Buffer(envsize, self.num_cycles, param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        self.iterations_per_target_update = param['q_learning']['iterations_per_target_update']
        self.iterations_since_last_target_update = 0
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
        
        self.ltl_lambda = param['lambda']
    
    def update_target_network(self):
        # copy current_network to target network
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp):
            with torch.no_grad():
                state_tensor = torch.tensor(state['mdp']).float().to(device)
                buchi = state['buchi']
                action, is_eps = self.Q.act(state_tensor, buchi)
            return action, is_eps
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state['mdp']).float().to(device)
                buchi = state['buchi']
                action, is_eps = self.Q.random_act(state, buchi)
            return action, is_eps

    def collect(self, s, b, a, r, lr, cr, s_, b_):
        if lr > 0:
            self.good_buffer.add(s, b, a, r, lr, cr, s_, b_)
        self.buffer.add(s, b, a, r, lr, cr, s_, b_)
    
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
    
    # def sample_from_both_buffers(self, batch_size):
    #     # first, sample (at most) half the data from the good buffer
    #     s, b, a, r, lr, cr, s_, b_ = self.good_buffer.sample(int(batch_size / 2))
    #     if len(s) < int(self.batch_size / 2):
    #         # if the good buffer doesn't have enough data, sample the rest from the bad buffer
    #         remaining_size = batch_size - len(s)
    #     else:
    #         remaining_size = int(batch_size / 2)
    #     s2, b2, a2, r2, lr2, cr2, s_2, b_2 = self.buffer.sample(remaining_size)
    #     s = np.concatenate((s, s2))
    #     b = np.concatenate((b, b2))
    #     a = np.concatenate((a, a2))
    #     r = np.concatenate((r, r2))
    #     lr = np.concatenate((lr, lr2))
    #     cr = np.concatenate((cr, cr2))
    #     s_ = np.concatenate((s_, s_2))
    #     b_ = np.concatenate((b_, b_2))
    #     return s, b, a, r, lr, cr, s_, b_
    
    def update(self):
        for _ in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                s, b, a, r, lr, cr, s_, b_ = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float).to(device)
                b = torch.tensor(b).type(torch.int64).unsqueeze(1).unsqueeze(1).to(device)
                s_ = torch.tensor(s_).type(torch.float).to(device)
                b_ = torch.tensor(b_).type(torch.int64).unsqueeze(1).unsqueeze(1).to(device)
                r = torch.tensor(r)
                lr = torch.tensor(lr)
                cr = torch.tensor(cr).to(device)
                a = torch.tensor(a).to(device)
                best_cycle_idx = torch.argmax(cr.sum(dim=0)).item()
                crewards = cr[:, best_cycle_idx]
                # if sum(crewards) > 0:
                #     import pdb; pdb.set_trace()
                targets = crewards + self.gamma * self.Q_target(s_, b_).amax(1)

            q_values = self.Q(s, b, False).gather(1, a.unsqueeze(1))
            # td_error = q_values - targets.to_tensor(0).unsqueeze(1).clone().detach()

            loss_func = torch.nn.SmoothL1Loss()
            loss = loss_func(q_values, targets.to_tensor(0).unsqueeze(1).clone().detach())
            
            if self.ltl_lambda != 0:
                normalized_val_loss = loss / (self.ltl_lambda / (1 - self.gamma))
            else:
                normalized_val_loss = loss

            # loss = (td_error**2).mean() # MSE

            # backward optimize
            self.optimizer.zero_grad()
            normalized_val_loss.backward()
            self.optimizer.step()

            
            # if loss.item() < 5:
            #     import pdb; pdb.set_trace()
            
            if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
                self.update_target_network()
                self.iterations_since_last_target_update = 0
        return loss.detach().mean()

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False, save_dir=None):
    states, buchis = [], []
    state, _ = env.reset()
    mdp_ep_reward = 0
    ltl_ep_reward = 0
    disc_ep_reward = 0
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    if not testing: agent.buffer.restart_traj()
    if testing & visualize:
        s = torch.tensor(state['mdp']).type(torch.float)
        b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
        #print(0, state['mdp'], state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
        #print(agent.Q(s, b))
    
    for t in range(1, param['q_learning']['T']):  # Don't infinite loop while learning
        action, is_eps = agent.select_action(state, testing)
        
        next_state, mdp_reward, done, info = env.step(action, is_eps)
        
        terminal = info['is_rejecting']
        constrained_reward, _, rew_info = env.constrained_reward(terminal, state['buchi'], next_state['buchi'], mdp_reward)

        if not testing: # TRAIN ONLY
            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, is_accepting = env.next_buchi(next_state['mdp'], buchi_state)
                    new_const_rew, _, new_rew_info = env.constrained_reward(terminal, buchi_state, next_buchi_state, mdp_reward)
                    agent.collect(state['mdp'], buchi_state, action, mdp_reward, new_rew_info["ltl_reward"], new_const_rew, next_state['mdp'], next_buchi_state)
                    if buchi_state == state['buchi']:
                        agent.buffer.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            next_buchi_state, is_accepting = env.next_buchi(state['mdp'], buchi_state, eps_idx)
                            new_const_rew, _, new_rew_info = env.constrained_reward(terminal, buchi_state, next_buchi_state, mdp_reward)
                            agent.collect(state['mdp'], buchi_state, action, mdp_reward, new_rew_info["ltl_reward"], new_const_rew, next_state['mdp'], next_buchi_state)
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], buchi_state, action, mdp_reward, rew_info["ltl_reward"], constrained_reward, next_state['mdp'], next_buchi_state)
                agent.buffer.mark()
        # agent.buffer.atomics.append(info['signal'])
        mdp_ep_reward += rew_info["mdp_reward"]
        ltl_ep_reward += rew_info["ltl_reward"]
        disc_ep_reward += param['gamma']**(t-1) * mdp_reward

        if done:
            break
        state = next_state
        states.append(state['mdp'])
        buchis.append(state['buchi'])
    if visualize:
        save_dir = save_dir + "/trajectory.png" if save_dir is not None else save_dir
        img = env.render(states=states, save_dir=save_dir)
        if img is not None:
            if testing: 
                runner.log({"testing": wandb.Image(img)})
        elif save_dir is not None:  # if we're in an environment that can't generate an image as in-python array
            # load image from save dir
        # frames = 
            if testing: 
                runner.log({"testing": wandb.Image(save_dir + "/trajectory.png")})
    # if ltl_ep_reward > 0:
    #     import pdb; pdb.set_trace()
    # if testing:
    #     import pdb; pdb.set_trace()
    return mdp_ep_reward, ltl_ep_reward, t
        
def run_Q_continuous(param, runner, env, second_order = False, visualize=True, save_dir=None):
    agent = Q_learning(env.observation_space, env.action_space, param['gamma'], env.num_cycles, param)
    fixed_state, _ = env.reset()
    for i_episode in tqdm(range(param['q_learning']['n_traj'])):
        # TRAINING
        mdp_ep_reward, ltl_ep_reward, t = rollout(env, agent, param, i_episode, runner, testing=False, save_dir=save_dir)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0 and i_episode != 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                mdp_test_reward, ltl_test_reward, t = rollout(env, agent, param, i_episode, runner, testing=True, visualize=visualize, save_dir=save_dir)
            #import pdb; pdb.set_trace()
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
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
                #  'TimestepsAlive': avg_timesteps,
                #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                    'ActionTemp': agent.temp,
                    #'EntropyLoss': loss_info["entropy_loss"],
                    "Test_R_LTL": ltl_test_reward,
                    "Test_R_MDP": mdp_test_reward
                    })
            
    plt.close()