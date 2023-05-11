import torch
import torch.nn as nn
import mtl
import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from utls.utls import STL_IDS, STLNode, parse_stl_into_tree
from policies.dqn import BufferSTL, DQNSTL, DQN
import wandb
from pathlib import Path
from tqdm import tqdm
import pandas as pd

device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#TODO: Either find or build out a better STL implementation with a parser, AST support, evaluation, etc.

class STL_Q_learning():
    def __init__(self, stl_tree, env_space, act_space, rho_alphabet, gamma, param) -> None:
        '''
        STL variant of Q-function, incorporating the recursive semantics of each STL operator and
        approximating a critic for each.
        param stl_tree: of type STLNode
        '''
        #super().__init__(env_space, act_space, gamma, param)
        # needs to be in parsable form - follow the above format for now with STLNodes
        self.stl_tree = stl_tree
        self.num_temporal_ops = self.set_ordering()
        self.outermost_negative = False
        if self.stl_tree.id == "~":  # outermost negative that we need to be considering in the policy!
            self.outermost_negative = True
        # shared parameters across all heads
        self.stl_q_net = DQNSTL(env_space, act_space, param, self.num_temporal_ops, self.outermost_negative)
        self.stl_q_target = DQNSTL(env_space, act_space, param, self.num_temporal_ops, self.outermost_negative)
        
        self.reward_q_net = DQN(env_space, act_space, param)
        self.reward_q_target = DQN(env_space, act_space, param)
        all_params = []
        for head in self.stl_q_net.heads:
            all_params.extend(head.parameters())
        all_params.extend(self.stl_q_net.actor.parameters())
        self.optimizer = torch.optim.Adam([
                {'params': all_params, 'lr': param['q_learning']['stl_lr']},
            ])

        self.reward_optimizer = torch.optim.Adam([
                {'params': self.reward_q_net.parameters(), 'lr': param['q_learning']['reward_lr']},
            ])


        self.update_target_network()
        self.rho_alphabet = rho_alphabet
        self.num_rho = len(self.rho_alphabet)
    
        self.buffer = BufferSTL(env_space['mdp'].shape, self.num_rho, param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        self.lambda_param = param['lambda']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        self.iterations_per_target_update = param['q_learning']['iterations_per_target_update']
        self.iterations_since_last_target_update = 0
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['total']
        self.td_error_vector = torch.zeros(self.num_temporal_ops, self.batch_size, self.n_mdp_actions)
        self.num_updates_called = 0
        
    
    def update_target_network(self):
        # copy current_network to target network
        self.stl_q_target.load_state_dict(self.stl_q_net.state_dict())
        self.reward_q_target.load_state_dict(self.reward_q_net.state_dict())
    
    # def select_action(self, state, is_testing):
    #     if is_testing or (np.random.uniform() > self.temp):
    #         with torch.no_grad():
    #             state_tensor = torch.FloatTensor(state['mdp']).to(device)
    #             buchi = state['buchi']
    #             action, is_eps, action_logprob = self.stl_q_net.act(state_tensor, buchi)

    #         return action, is_eps, action_logprob
    #     else:
    #         with torch.no_grad():
    #             state_tensor = torch.FloatTensor(state['mdp']).to(device)
    #             buchi = state['buchi']
    #             action, is_eps, action_logprob = self.stl_q_net.random_act(state, buchi)
    #         return action, is_eps, action_logprob
    
    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                buchi = state['buchi']
                stl_action, is_eps, action_logprob, stl_qs = self.stl_q_net.act(state_tensor, buchi)
                rew_action, is_eps, action_logprob, rew_qs = self.reward_q_net.act(state_tensor, buchi)
                combined_qs = (1 - self.lambda_param) * rew_qs + self.lambda_param * stl_qs
                new_action = int(combined_qs.argmax())
                new_is_eps = new_action > self.n_mdp_actions
            return new_action, new_is_eps, action_logprob
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                buchi = state['buchi']
                action, is_eps, action_logprob = self.stl_q_net.random_act(state, buchi)
            return action, is_eps, action_logprob

    def collect(self, s, b, a, rew, r, s_, b_):
        self.buffer.add(s, b, a, rew, r, s_, b_)
    
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
        
    def reset_td_errors(self):
        self.td_error_vector = torch.zeros(self.num_temporal_ops, self.batch_size, requires_grad=False)
        
    def set_ordering(self):
        num_expr = 0
        queue = [self.stl_tree]
        num_temporal_ops = 0
        while len(queue) > 0:
            curr = queue.pop(0)
            if curr.id != "rho":
                num_expr += 1
                # set the head that'll correspond to this operator
                if curr.id in ["G", "E", "F", "X"]:
                    curr.set_ordering(num_temporal_ops)
                    num_temporal_ops += 1
            for child in curr.children:
                queue.append(child)
        return num_temporal_ops
    
    def recurse_node(self, current_node, s, b, act, rhos, s_next, b_next):
        cid = current_node.id
        if cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = rhos[:, self.rho_alphabet.index(current_node.rho)]
            return phi_val
        elif cid in ["&", "|"]:
            all_phi_vals = []
            for child in current_node.children:
                all_phi_vals.append(self.recurse_node(child, s, b, act, rhos, s_next, b_next))
            # and case and or case are min and max, respectively
            phi_val = torch.min(torch.stack(all_phi_vals), dim=0).values if cid == "&" else torch.max(torch.stack(all_phi_vals), dim=0).values
            return phi_val
        elif cid == "~":  # negation case
            phi_val = self.recurse_node(current_node.children[0], s, b, act, rhos, s_next, b_next)
            return -1 * phi_val
        else:  # G or E case: just get it by recursing with a single child
            phi_val = self.recurse_node(current_node.children[0], s, b, act, rhos, s_next, b_next)
        
        
        # q_action = q_s_next_head[torch.arange(q_s_next_head.shape[0]), act]  # TODO: is there a smarter way to do this?

        ## originally: Q(s) ~=   min(r, gamma * max_{a'} Q(s', a')) 
        ## ours:       Q(s) ~=       r + gamma * max_{a'} Q(s', a')
        Qs = self.stl_q_target.interior_forward(s_next, b_next, current_node.order)
        
        q_action = Qs.to_tensor(0)[torch.arange(s_next.shape[0]), act]
        if cid == "G":
            td_val = torch.minimum(phi_val, self.gamma * q_action)
            self.td_error_vector[current_node.order, :] = td_val.float()
        elif cid in ["E", "F"] :
            td_val = torch.maximum(phi_val, self.gamma * q_action)
            self.td_error_vector[current_node.order, :] = td_val.float()
        elif cid == "X":
            td_val = self.gamma * q_action
            self.td_error_vector[current_node.order] = td_val
        return phi_val
        
    
    def update(self, runner):
        num_prev_epochs = self.num_updates_called * self.n_batches
        self.num_updates_called += 1

        # update based on the STL recursive semantics
        #TODO: check that our recursive traversal does the same
        # as the DFS search in set_heads
        #TODO: finish
        total_loss = 0
        loss_func = torch.nn.SmoothL1Loss()
        total_reward_loss = 0
        for k in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                # s, a, rhos, s_, = self.buffer.sample(self.batch_size)
                
                s, b, a, rew, rhos, s_, b_ = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float)
                b = torch.tensor(b).type(torch.int64).unsqueeze(1).unsqueeze(1)
                s_ = torch.tensor(s_).type(torch.float)
                b_ = torch.tensor(b_).type(torch.int64).unsqueeze(1).unsqueeze(1)
                rews = torch.tensor(rew)
                rhos = torch.tensor(rhos)
                a = torch.tensor(a)
                
                # compute TD values
                self.reset_td_errors()
                Q_rew = self.reward_q_target(s_, b_)
                reward_targets = rews + self.gamma * Q_rew.to_tensor(0).amax(axis=1)
                Qs = self.stl_q_target(s_, b_)
                if self.stl_q_target.outermost_negative:
                    Qs *= -1
                max_actions = Qs.argmax(dim = 1).to_tensor(0).long()
                self.recurse_node(self.stl_tree, s, b, max_actions, rhos, s_, b_)
            
            # compute all q values for all heads
            # q_values = []  ##  TODO: check if this is ok
            # q_val_base = self.stl_q_net.forward_base(s)
            # for head_idx in range(self.num_temporal_ops):
            #     #q_values.append(self.stl_q_net.forward_head(s, head_idx))
            #     q_values.append(self.stl_q_net.forward_head(q_val_base, head_idx))
            # q_values = torch.stack(q_values)
            
            # TODO: Make more computationally efficient by not calling the base so many times

            # Q(s, a)
            q_values = torch.stack([self.stl_q_net.interior_forward(s, b, head_idx, False)[torch.arange(len(a)), a] for head_idx in range(self.num_temporal_ops)])
            with torch.no_grad():
                #  ~= r + gamma * max_{a'} Q(s', a')
                targets = self.td_error_vector.clone().detach() #[:, torch.arange(len(a)), a]

            q_reward_values = self.reward_q_net(s, b, False).gather(1, a.unsqueeze(1)).squeeze()

            # print(self.stl_q_target(torch.tensor(self.buffer.states[self.buffer.current_traj]).float(), torch.tensor(self.buffer.buchis[self.buffer.current_traj]).type(torch.int64).unsqueeze(1).unsqueeze(1)))
            # print(targets.min(), targets.max())
            # print(q_values.min(), q_values.max())
            # print()
            loss = loss_func(q_values, targets.clone().detach())            
            reward_loss = loss_func(q_reward_values, reward_targets.clone().detach())
            total_loss += loss
            total_reward_loss += reward_loss
            # backward optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # optimize for the reward net as well
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()
            runner.log({
                'stl_loss': loss.item(),
                'reward_loss': reward_loss.item()
            },
                step=num_prev_epochs + k)

        if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
            print('Updating Params' + '*'*50)
            self.update_target_network()
            self.iterations_since_last_target_update = 0
        return total_loss / self.n_batches       

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False):
    state, _ = env.reset()
    states = []
    if not testing: agent.buffer.restart_traj()
    states.append(state['mdp'])

    # actions = [0 if j < 6 else 4 for j in range(param['q_learning']['T'])]
    # actions = [0 for j in range(param['q_learning']['T'])]

    for t in range(param['q_learning']['T']):  # Don't infinite loop while learning
        action, is_eps, log_prob = agent.select_action(state, testing)

        # action = actions[t]
        next_state, cost, done, info = env.step(action, is_eps)
        # reward = info['is_accepting']
        rhos = info['rho']

        if not testing: 
            agent.collect(state['mdp'], state['buchi'], action, cost, rhos, next_state['mdp'], next_state['buchi'])
            agent.buffer.mark()

        if done:
            break
        state = next_state
        states.append(next_state['mdp'])

    all_rho_vals = env.mdp.episode_rhos
    if visualize:
        # frames = 
        # runner.log({"video": wandb.Video([env.render(states=np.atleast_2d(state), save_dir=None) for state in states], fps=10)})
        if testing: 
            runner.log({"testing": wandb.Image(env.render(states=states, save_dir=None))})
        else:
            runner.log({"training": wandb.Image(env.render(states=states, save_dir=None))})
    return all_rho_vals, 0, t

def run_Q_STL(param, runner, env):
    stl_tree = parse_stl_into_tree(param['ltl']['formula'])
    # varphi = mtl.parse(param['ltl']['formula'])
    agent = STL_Q_learning(stl_tree, env.observation_space, env.action_space, env.mdp.rho_alphabet, param['gamma'], param)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()
    current_loss = 0  # placeholder value
    checkpoint_count = 0
    all_losses = [current_loss]

    for i_episode in tqdm(range(param['q_learning']['n_traj'])):
        # TRAINING
        all_rhos, max_q_val, t = rollout(env, agent, param, i_episode, runner, testing=False, visualize=((i_episode % 50) == 0))
        stl_val = -100 #varphi(all_rhos)

        # Qs = agent.stl_q_net(torch.tensor(agent.buffer.states[agent.buffer.current_traj]).float(), torch.tensor(agent.buffer.buchis[agent.buffer.current_traj]).type(torch.int64).unsqueeze(1).unsqueeze(1))
        # pd.DataFrame(env.mdp.episode_rhos['y'])
        # import pdb; pdb.set_trace()
        # if stl_val > 0:
        #     print(i_episode, agent.buffer.get_rhos())
        #     X = torch.tensor(agent.buffer.states[agent.buffer.current_traj])
        #     A = torch.tensor(agent.buffer.buchis[agent.buffer.current_traj])
        #     agent.stl_q_net.forward(X.type(torch.float), A.type(torch.int64).unsqueeze(1).unsqueeze(1)).argmax(1)
        #     agent.buffer.actions[agent.buffer.current_traj]
        #     agent.buffer.actions[agent.buffer.current_traj]
        #     import pdb; pdb.set_trace()

        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update(runner).item()
            all_losses.append(current_loss)
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                all_rhos, max_q_val, t = rollout(env, agent, param, test_iter, runner, testing=True, visualize= ((i_episode % 50) == 0) & (test_iter == 0) )
                stlval = varphi(all_rhos)
                test_data.append((stlval, t))
            test_data = np.array(test_data)

        # if i_episode % param['checkpoint_freq__n_episodes'] == 0:
        #     ckpt_path = Path(runner.dir) / f'checkpoint_{checkpoint_count}.pt'
        #     torch.save(agent.state_dict(), ckpt_path)
        #     artifact = wandb.Artifact('checkpoint', type='model')
        #     artifact.add_file(ckpt_path)
        #     runner.log_artifact(artifact)
        #     checkpoint_count += 1
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [stl_val]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            #import pdb; pdb.set_trace()
            success_history += [test_data[:, 0].mean()]
            # method = 'STLQ'
            # plot_something_live(axes, [np.arange(len(history)),  np.arange(len(all_losses))], [history, all_losses], method)
            # logger.logkv('Iteration', i_episode)
            # logger.logkv('Method', method)
            # logger.logkv('Success', success_history[-1])
            # logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            # logger.logkv('EpisodeRhos', stl_val)
            # logger.logkv('ExpectedQVal', max_q_val)  
            # logger.logkv('LossVal', current_loss)
            # logger.logkv('TimestepsAlive', avg_timesteps)
            # logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['q_learning']['T'])
            # logger.logkv('ActionTemp', agent.temp)
            
            # logger.dumpkvs()
            runner.log({'Iteration': i_episode,
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
                    'STL': stl_val,
                    'LossVal': current_loss,
                    #  'TimestepsAlive': avg_timesteps,
                    #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                     'ActionTemp': agent.temp,
                     })
            
    plt.close()