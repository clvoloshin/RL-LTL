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
from PIL import Image


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
                        {'params': self.Q.actor.parameters(), 'lr': param['q_learning']['lr'], 'betas': (0.9, 0.9)},
                    ])

        self.update_target_network()
        self.num_cycles = num_cycles
        self.buffer = Buffer(envsize, self.num_cycles, param['replay_buffer_size'])
        self.good_buffer = Buffer(envsize, self.num_cycles, param['replay_buffer_size'])
        self.init_temp = param['q_learning']['init_temp']
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

    def save_model(self, save_path):
        torch.save(self.Q.state_dict(), save_path)
    
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
        # print(f'Setting temperature: {self.temp}')
    
    def reset_entropy(self):
        self.temp = self.init_temp
    
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
        for _ in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                s, b, a, r, lr, cr, s_, b_ = self.buffer.sample(self.batch_size)
                #s, b, a, r, lr, cr, s_, b_ = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float).to(device)
                b = torch.tensor(b).type(torch.int64).unsqueeze(1).unsqueeze(1).to(device)
                s_ = torch.tensor(s_).type(torch.float).to(device)
                b_ = torch.tensor(b_).type(torch.int64).unsqueeze(1).unsqueeze(1).to(device)
                r = torch.tensor(r)
                lr = torch.tensor(lr)
                cr = torch.tensor(cr).to(device)
                a = torch.tensor(a).to(device)
                best_cycle_idx = torch.argmax(lr.sum(dim=0)).item()
                crewards = cr[:, best_cycle_idx].float()
                targets = crewards + self.gamma * self.Q_target(s_, b_).amax(1)

            q_values = self.Q(s, b, False).gather(1, a.unsqueeze(1))
            # td_error = q_values - targets.to_tensor(0).unsqueeze(1).clone().detach()
            #import pdb; pdb.set_trace()
            loss_func = torch.nn.MSELoss()
            loss = loss_func(q_values, targets.to_tensor(0).unsqueeze(1).clone().detach())
            
            if self.ltl_lambda != 0:
                normalized_val_loss = loss / (self.ltl_lambda / (1 - self.gamma))
            else:
                normalized_val_loss = loss

            # backward optimize
            self.optimizer.zero_grad()
            normalized_val_loss.backward()
            self.optimizer.step()

            
            if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
                self.update_target_network()
                self.iterations_since_last_target_update = 0
        return loss.detach().mean()

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
    ltl_rewards = []
    sum_xformed_rewards = np.zeros(env.num_cycles)
    
    for t in range(1, param['q_learning']['T']):  # Don't infinite loop while learning
        action, is_eps = agent.select_action(state, testing)
        
        next_state, mdp_reward, done, info = env.step(action, is_eps)
        
        terminal = info['is_rejecting']
        constrained_reward, _, rew_info = env.constrained_reward(terminal, state['buchi'], next_state['buchi'], mdp_reward, info["rhos"])

        if not testing: # TRAIN ONLY
            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, is_accepting = env.next_buchi(next_state['mdp'], buchi_state)
                    new_const_rew, _, new_rew_info = env.constrained_reward(terminal, buchi_state, next_buchi_state, mdp_reward, info["rhos"])
                    agent.collect(state['mdp'], buchi_state, action, mdp_reward, new_rew_info["ltl_reward"], new_const_rew, next_state['mdp'], next_buchi_state)
                    if buchi_state == state['buchi']:
                        agent.buffer.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            next_buchi_state, is_accepting = env.next_buchi(state['mdp'], buchi_state, eps_idx)
                            new_const_rew, _, new_rew_info = env.constrained_reward(terminal, buchi_state, next_buchi_state, mdp_reward, info["rhos"])
                            agent.collect(state['mdp'], buchi_state, action, mdp_reward, new_rew_info["ltl_reward"], new_const_rew, next_state['mdp'], next_buchi_state)
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], buchi_state, action, mdp_reward, rew_info["ltl_reward"], constrained_reward, next_state['mdp'], next_buchi_state, info["rhos"])
                agent.buffer.mark()
        visit_buchi = next_state['buchi'] in env.automaton.automaton.accepting_states
        ltl_rewards.append(rew_info["ltl_reward"])
        sum_xformed_rewards += rew_info["ltl_reward"]
        mdp_ep_reward += rew_info["mdp_reward"]
        ltl_ep_reward += max(rew_info["ltl_reward"])
        constr_ep_reward += (agent.ltl_lambda * (visit_buchi) + mdp_reward)
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
    ltl_ep_reward = np.array(ltl_rewards).sum(axis=0)[np.argmax(sum_xformed_rewards)]
    return mdp_ep_reward, ltl_ep_reward, constr_ep_reward, total_buchi_visits, img, np.array(buchi_visits), np.array(mdp_rewards)
        
def run_Q_continuous(param, runner, env, second_order = False, visualize=True, save_dir=None, save_model=True, agent=None, n_traj=None):
    if agent is None:
        agent = Q_learning(env.observation_space, env.action_space, param['gamma'], env.num_cycles, param)
    fixed_state, _ = env.reset()
    best_creward = -1 * float('inf')
    all_crewards = []
    all_bvisit_trajs = []
    all_mdpr_trajs = []
    all_test_bvisits = []
    all_test_mdprs = []
    all_test_crewards = []
    test_creward = 0
    best_agent = None
    if n_traj is None:
        n_traj = param['q_learning']['n_traj']
    for i_episode in tqdm(range(n_traj)):
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, save_dir=save_dir)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0 and i_episode != 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        all_crewards.append(creward)
        all_bvisit_trajs.append(bvisit_traj)
        all_mdpr_trajs.append(mdp_traj)
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_avg_bvisits = 0
            test_avg_mdp_reward = 0
            test_avg_ltl_reward = 0
            test_avg_creward = 0
            for trial in range(param['testing']['num_rollouts']):
                mdp_test_reward, ltl_test_reward, test_creward, bvisits, img, test_bvisit_traj, test_mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, visualize=visualize, save_dir=save_dir) #param['n_traj']-100) ))
                test_avg_bvisits += bvisits
                test_avg_mdp_reward += mdp_test_reward
                test_avg_ltl_reward += ltl_test_reward
                test_avg_creward += test_creward
            test_avg_bvisits /= param['testing']['num_rollouts']
            test_avg_mdp_reward /= param['testing']['num_rollouts']
            test_avg_ltl_reward /= param['testing']['num_rollouts']
            test_avg_creward /= param['testing']['num_rollouts']
            test_creward = test_avg_creward
            if test_creward >= best_creward and save_model:
                best_creward = test_creward
                best_agent = agent.Q.state_dict()
                agent.save_model(save_dir + "/" + param["model_name"] + ".pt")
            all_test_bvisits.append(test_avg_bvisits)
            all_test_mdprs.append(test_avg_mdp_reward)
            all_test_crewards.append(test_creward)
            testing = True
        else:
            testing = False

        if visualize and testing:
            if i_episode >= 1 and i_episode % 1 == 0:
                if img is None: 
                    to_log = save_dir + "/trajectory.png" if save_dir is not None else save_dir
                else:
                    to_log = img
                runner.log({'R_LTL': ltl_ep_reward,                
                            'R_MDP': mdp_ep_reward,
                            'LossVal': current_loss,
                            'ActionTemp': agent.temp,
                            "Test_R_LTL": test_avg_ltl_reward,
                            "Test_R_MDP": test_avg_mdp_reward,
                            "Buchi_Visits": test_avg_bvisits,
                            "Dual Reward": test_avg_creward,
                            "testing": wandb.Image(to_log)})
        else:
            if i_episode % 1 == 0:
                runner.log({#'Iteration': i_episode,
                    'R_LTL': ltl_ep_reward,
                    'R_MDP': mdp_ep_reward,
                    'LossVal': current_loss,
                    'ActionTemp': agent.temp,
                    "Test_R_LTL": test_avg_ltl_reward,
                    "Test_R_MDP": test_avg_mdp_reward,
                    "Buchi_Visits": test_avg_bvisits,
                    "Dual Reward": test_avg_creward,

                        })
    # import pdb; pdb.set_trace()
    agent.Q.load_state_dict(best_agent)     
    return agent, all_crewards, all_bvisit_trajs, all_mdpr_trajs, all_test_crewards, all_test_bvisits, all_test_mdprs
    
def eval_q_agent(param, env, agent, visualize=True, save_dir=None):
    if agent is None:
        agent = Q_learning(
        env.observation_space, 
        env.action_space, 
        param['gamma'], 
        env.num_cycles,
        param, 
        model_path=param['load_path'])
    fixed_state, _ = env.reset()
    crewards = []
    mdp_rewards = []
    avg_buchi_visits = []
    print("Beginning evaluation.")
    for i_episode in tqdm(range(param["num_eval_trajs"])):
        img_path = save_dir + "/eval_traj_{}.png".format(i_episode) if save_dir is not None else save_dir
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, None, testing=False, visualize=visualize, save_dir=img_path, eval=True)
        mdp_rewards.append(mdp_ep_reward)
        avg_buchi_visits.append(bvisits)
        crewards.append(creward)
        if img is not None:
            im = Image.fromarray(img)
            if img_path is not None:
                im.save(img_path)
    mdp_test_reward, ltl_test_reward, test_creward, test_bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, None, testing=True, visualize=visualize)
    print("Buchi Visits and MDP Rewards for fixed (test) policy at Eval Time:")
    print("Buchi Visits:", test_bvisits)
    print("MDP Reward:", mdp_test_reward)
    print("")
    print("Average Buchi Visits and Average MDP Rewards for stochastic policy at Eval Time:")
    print("Buchi Visits:", sum(avg_buchi_visits) / len(avg_buchi_visits))
    print("MDP Reward:", sum(mdp_rewards) / len(mdp_rewards))
    return test_bvisits, mdp_test_reward, sum(avg_buchi_visits) / len(avg_buchi_visits), sum(mdp_rewards) / len(mdp_rewards), sum(crewards) / len(crewards)