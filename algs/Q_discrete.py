import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tqdm import tqdm
from PIL import Image
import wandb

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
import numpy.ma as ma
import pandas as pd

class Buffer:
    def __init__(self, num_cycles, max_ = 10000) -> None:
        self.max_ = max_
        self.counter = -1
        self.num_cycles = num_cycles

        self.states = np.array([0 for _ in range(max_)])
        self.actions = np.array([0 for _ in range(max_)])
        self.rewards = np.array([0 for _ in range(max_)])
        self.ltl_rewards = np.zeros((max_, self.num_cycles))
        self.cycle_rewards = np.zeros((max_, self.num_cycles))
        self.next_states = np.array([0 for _ in range(max_)])
        self.buchis = np.array([0 for _ in range(max_)])
        self.next_buchis = np.array([0 for _ in range(max_)])

        self.current_traj = []
        self.all_current_traj = []
    
    def add(self, s, b, a, r, lr, cr, s_, b_):
        self.counter += 1
        self.states[self.counter % self.max_] = s
        self.buchis[self.counter % self.max_] = b
        self.next_states[self.counter % self.max_] = s_
        self.next_buchis[self.counter % self.max_] = b_
        self.actions[self.counter % self.max_] = a
        self.rewards[self.counter % self.max_] = r
        self.ltl_rewards[self.counter % self.max_] = lr
        self.cycle_rewards[self.counter % self.max_] = cr
        self.all_current_traj.append(self.counter % self.max_)
    
    def mark(self):
        self.current_traj.append(self.counter % self.max_)
    
    def restart_traj(self):
        self.current_traj = []
        self.all_current_traj = []
    
    def get_current_traj(self):
        idxs = np.array(self.current_traj)
        df = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_']

        idxs = np.array(self.all_current_traj)
        df2 = pd.DataFrame([self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]]).T
        df2.columns = ['s', 'b', 'a', 'r', 'lr', 'cr', 's_', 'b_']
        return df, df2
    
    def get_all(self):
        idxs = np.arange(0, min(self.counter, self.max_-1))
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]

    def sample(self, batchsize):
        if self.counter < batchsize:
            # import pdb; pdb.set_trace()
            return self.states, self.buchis, self.actions, self.rewards, self.ltl_rewards, self.cycle_rewards, self.next_states, self.next_buchis
        idxs = np.random.random_integers(0, min(self.counter, self.max_-1), batchsize)
        return self.states[idxs], self.buchis[idxs], self.actions[idxs], self.rewards[idxs], self.ltl_rewards[idxs], self.cycle_rewards[idxs], self.next_states[idxs], self.next_buchis[idxs]


class Q_learning:
    def __init__(self, env_space, act_space, gamma, num_cycles, param) -> None:
        self.num_cycles = num_cycles
        self.good_buffer = Buffer(self.num_cycles, param['replay_buffer_size'])

        self.buffer = Buffer(self.num_cycles, param['replay_buffer_size'])
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
        return act, is_eps
    
    def collect(self, s, b, a, r, lr, cr, s_, b_):
        if max(lr) > 0:
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
    
    def sample_from_both_buffers(self, batch_size):
        # first, sample (at most) half the data from the good buffer
        s, b, a, r, lr, cr, s_, b_ = self.good_buffer.sample(int(batch_size / 2))
        if len(s) < int(self.batch_size / 2):
            # if the good buffer doesn't have enough data, sample the rest from the bad buffer
            remaining_size = batch_size - len(s)
        else:
            remaining_size = int(batch_size / 2)
        s2, b2, a2, r2, lr2, cr2, s_2, b_2 = self.buffer.sample(remaining_size)
        s = np.concatenate((s, s2))
        b = np.concatenate((b, b2))
        a = np.concatenate((a, a2))
        r = np.concatenate((r, r2))
        lr = np.concatenate((lr, lr2))
        cr = np.concatenate((cr, cr2))
        s_ = np.concatenate((s_, s_2))
        b_ = np.concatenate((b_, b_2))
        return s, b, a, r, lr, cr, s_, b_
    
    def update(self):
        # for _ in range(self.n_batches):
        #     s, b, a, r, s_, b_ = self.buffer.get_all()
        #     self.Q[s.astype(int), b.astype(int), a.astype(int)] = r + self.gamma * self.Q[s_.astype(int), b_.astype(int)].max(axis=1)
        for _ in range(self.n_batches):
            s, b, a, r, lr, cr, s_, b_ = self.sample_from_both_buffers(self.batch_size)
            best_cycle_idx = np.argmax(lr.sum(axis=0)).item()
            crewards = cr[:, best_cycle_idx]
            self.Q[s.astype(int), b.astype(int), a.astype(int)] = crewards + self.gamma * self.Q[s_.astype(int), b_.astype(int)].max(axis=1)

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False, save_dir=None, eval=False):
    states, buchis = [], []
    state, _ = env.reset()
    mdp_ep_reward = 0
    ltl_ep_reward = 0
    disc_ep_reward = 0
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    constr_ep_reward = 0
    total_buchi_visits = 0
    if not testing: 
        agent.buffer.restart_traj()
    buchi_visits = []
    mdp_rewards = []
    # if testing & visualize:
    #     s = torch.tensor(state['mdp']).type(torch.float)
    #     b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
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
        visit_buchi = next_state['buchi'] in env.automaton.automaton.accepting_states
        mdp_ep_reward += rew_info["mdp_reward"]
        ltl_ep_reward += max(rew_info["ltl_reward"])
        disc_ep_reward += param['gamma']**(t-1) * mdp_reward
        buchi_visits.append(visit_buchi)
        mdp_rewards.append(mdp_reward)
        total_buchi_visits += visit_buchi
        if done:
            break
        state = next_state
        states.append(state['mdp'])
        buchis.append(state['buchi'])
    if visualize:
        if eval:
            save_dir = save_dir
        else:
            save_dir = save_dir + "/trajectory.png" if save_dir is not None else save_dir
        img = env.render(states=states, save_dir=save_dir)
    else:
        img = None
    # if ltl_ep_reward > 0:
    #     import pdb; pdb.set_trace()
    # if testing:
    #     import pdb; pdb.set_trace()
    return mdp_ep_reward, ltl_ep_reward, constr_ep_reward, total_buchi_visits, img, np.array(buchi_visits), np.array(mdp_rewards)

        
def run_Q_discrete(param, runner, env, second_order = False, visualize=True, save_dir=None, agent=None):
    if agent is None:
        agent = Q_learning(env.observation_space, env.action_space, param['gamma'], env.num_cycles, param)
    fixed_state, _ = env.reset()
    best_creward = 0
    all_crewards = []
    all_bvisit_trajs = []
    all_mdpr_trajs = []
    test_creward = 0
    # import pdb; pdb.set_trace()
    for i_episode in tqdm(range(param['q_learning']['n_traj'])):
        # TRAINING
        # if i_episode > 300:
        #     import pdb; pdb.set_trace()
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, save_dir=save_dir)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0 and i_episode != 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            mdp_test_reward, ltl_test_reward, test_creward, bvisits, img, test_bvisit_traj, test_mdp_traj = rollout(env, agent, param, i_episode, runner, testing=True, visualize=visualize, save_dir=save_dir)
            testing = True
        else:
            testing = False
            #import pdb; pdb.set_trace()
        # if img is not None:
        #     if testing: 
        #         runner.log({"testing": wandb.Image(img)})
        # elif save_dir is not None:  # if we're in an environment that can't generate an image as in-python array
        #     # load image from save dir
        # # frames = 
        #     if testing: 
        #         runner.log({"testing": wandb.Image(save_dir + "/trajectory.png")})
        all_crewards.append(creward)
        all_bvisit_trajs.append(bvisit_traj)
        all_mdpr_trajs.append(mdp_traj)
        if visualize and testing:
            if i_episode >= 1 and i_episode % 1 == 0:
                if img is None: 
                    to_log = save_dir + "/trajectory.png" if save_dir is not None else save_dir
                else:
                    to_log = img
                runner.log({'R_LTL': ltl_ep_reward,                
                            'R_MDP': mdp_ep_reward,
                            'LossVal': current_loss,
                            #'AvgTimesteps': t,
                            #  'TimestepsAlive': avg_timesteps,
                            #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                                'ActionTemp': agent.temp,
                                #'EntropyLoss': loss_info["entropy_loss"],
                                "Test_R_LTL": ltl_test_reward,
                                "Test_R_MDP": mdp_test_reward,
                                "Dual Reward": test_creward,
                                "BVisits": bvisits,
                                "testing": wandb.Image(to_log)})
            # runner.log({'R_LTL': ltl_ep_reward,
            #     'R_MDP': mdp_ep_reward,
            #     'LossVal': current_loss,
            #     #'AvgTimesteps': t,
            #     #  'TimestepsAlive': avg_timesteps,
            #     #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
            #         'ActionTemp': agent.temp,
            #         #'EntropyLoss': loss_info["entropy_loss"],
            #         "Test_R_LTL": ltl_test_reward,
            #         "Test_R_MDP": mdp_test_reward
            #         "testing": wandb.Image(to_log)})
        else:
            if i_episode % 1 == 0:
                runner.log({#'Iteration': i_episode,
                    'R_LTL': ltl_ep_reward,
                    'R_MDP': mdp_ep_reward,
                    'LossVal': current_loss,
                    #'AvgTimesteps': t,
                    #  'TimestepsAlive': avg_timesteps,
                    #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                        'ActionTemp': agent.temp,
                        #'EntropyLoss': loss_info["entropy_loss"],
                        "Test_R_LTL": ltl_test_reward,
                        "Test_R_MDP": mdp_test_reward,
                        "BVisits": bvisits,
                        "Dual Reward": test_creward,

                        })
                
    return agent, all_crewards, all_bvisit_trajs, all_mdpr_trajs
    
def eval_q_agent(param, runner, env, agent, visualize=True, save_dir=None):
    if agent is None:
        agent = Q_learning(
        env.observation_space, 
        env.action_space, 
        param['gamma'], 
        env.num_cycles,
        param, 
        model_path=param['load_path'])
    fixed_state, _ = env.reset()
    #runner.log({"testing": wandb.Image(env.render(states=[fixed_state['mdp']], save_dir=None))})
    crewards = []
    mdp_rewards = []
    avg_buchi_visits = []
    print("Beginning evaluation.")
    for i_episode in tqdm(range(param["num_eval_trajs"])):
        img_path = save_dir + "/eval_traj_{}.png".format(i_episode) if save_dir is not None else save_dir
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, visualize=visualize, save_dir=img_path, eval=True)
        mdp_rewards.append(mdp_ep_reward)
        avg_buchi_visits.append(bvisits)
        crewards.append(creward)
        if img is not None:
            im = Image.fromarray(img)
            if img_path is not None:
                im.save(img_path)
    mdp_test_reward, ltl_test_reward, test_creward, test_bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, runner, testing=True, visualize=visualize)
    print("Buchi Visits and MDP Rewards for fixed (test) policy at Eval Time:")
    print("Buchi Visits:", test_bvisits)
    print("MDP Reward:", mdp_test_reward)
    print("")
    print("Average Buchi Visits and Average MDP Rewards for stochastic policy at Eval Time:")
    print("Buchi Visits:", sum(avg_buchi_visits) / len(avg_buchi_visits))
    print("MDP Reward:", sum(mdp_rewards) / len(mdp_rewards))
    return sum(avg_buchi_visits) / len(avg_buchi_visits), sum(mdp_rewards) / len(mdp_rewards), sum(crewards) / len(crewards)