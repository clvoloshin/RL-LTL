import torch
import torch.nn as nn
from Q_continuous import Q_learning

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from policies.dqn import BufferStandard, DQNSTL

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
#TODO: Either find or build out a better STL implementation with a parser, AST support, evaluation, etc.
STL_IDS = ["G", "E", "&", "~", "or", "rho"]

class STLNode():
    
    def __init__(self, id: str, children, time_bounds: tuple=None, head=None, rho=None) -> None:
        self.id = id
        self.children = children
        if self.id == "rho":
            
            assert self.children == None
        if self.id not in ["&", "or"]:
            # if it's not an 'and' or an 'or', it should only have one child
            assert len(self.children) == 1
        self.rho = rho
        self.time_bounds = time_bounds
        self.head = head
        self.order = None
    
    def set_ordering(self, order):
        self.order = order
    

class STL_Q_learning():
    def __init__(self, stl_tree, env_space, act_space, gamma, param) -> None:
        '''
        STL variant of Q-function, incorporating the recursive semantics of each STL operator and
        approximating a critic for each.
        param stl_tree: of type STLNode
        '''
        # inherit the original q-learning methods
        super().__init__(env_space, act_space, gamma, param)
        # needs to be in parsable form - follow the above format for now with STLNodes
        self.stl_tree = stl_tree
        self.num_temporal_ops = self.set_ordering()
        # shared parameters across all heads
        self.stl_q_net = DQNSTL(env_space, act_space, param, self.num_temporal_ops)
        self.stl_q_target = DQNSTL(env_space, act_space, param, self.num_temporal_ops)
        
        self.optimizer = torch.optim.Adam([
                {'params': self.Q.actor.parameters(), 'lr': param['q_learning']['lr']},
            ])

        self.update_target_network()
    
        self.buffer = BufferStandard(env_space['mdp'].shape, param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        self.iterations_per_target_update = param['q_learning']['iterations_per_target_update']
        self.iterations_since_last_target_update = 0
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
        self.td_error_vector = torch.zeros(self.num_temporal_ops, self.n_mdp_actions)

    
    def update_target_network(self):
        # copy current_network to target network
        self.stl_q_target.load_state_dict(self.stl_q_net.state_dict())

    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                action = self.Q.act(state_tensor)

            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state['mdp']).to(device)
                action = self.Q.random_act(state_tensor)
            return action
    
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
        self.td_error_vector = torch.zeros(self.num_temporal_ops)
        
    def collect(self, s, b, a, r, s_, b_):
        self.buffer.add(s, b, a, r, s_, b_)
    
    def set_ordering(self):
        num_expr = 0
        queue = [self.stl_tree]
        num_temporal_ops = 0
        while len(queue) > 0:
            curr = queue.pop()
            if curr.id != "rho":
                num_expr += 1
                # set the head that'll correspond to this operator
                if curr.id in ["G", "E"]:
                    curr.set_order(num_temporal_ops)
                    num_temporal_ops += 1
            for child in curr.children:
                queue.append(child)
        return num_temporal_ops
    
    def recurse_node(self, current_node, s, a, s_next):
        cid = current_node.id
        if cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = current_node.rho(s)
            return phi_val
        elif cid in ["&", "or"]:
            all_phi_vals = []
            for child in current_node.children:
                all_phi_vals.append(self.recurse_node(child, s, a, s_next))
            # and case and or case are min and max, respectively
            phi_val = torch.min(all_phi_vals) if cid == "&" else torch.max(all_phi_vals)
            return phi_val
        elif cid == "~":  # negation case
            phi_val = self.recurse_node(current_node.children[0], s, a, s_next)
            return -1 * phi_val
        else:  # G or E case: just get it by recursing with a single child
            phi_val = self.recurse_node(current_node.children[0], s, a, s_next)
        v_s_next = self.stl_q_net.forward_base(s_next)
        # operator-specific head
        v_s_next_head = self.stl_q_net.forward_head(v_s_next, current_node.order)
        if cid == "G":
            td_val = torch.min(phi_val, self.gamma * v_s_next_head)
            self.td_error_vector[current_node.order] = td_val
        elif cid == "E":
            td_val = torch.min(phi_val, self.gamma * v_s_next_head)
            self.td_error_vector[current_node.order] = td_val
        return phi_val
        
    
    def update(self):
        # update based on the STL recursive semantics
        #TODO: check that our recursive traversal does the same
        # as the DFS search in set_heads
        #TODO: finish
        for _ in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                s, a, r, s_, = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float)
                s_ = torch.tensor(s_).type(torch.float)
                r = torch.tensor(r)
                a = torch.tensor(a)
                # compute TD values
                self.recurse_node(self.stl_tree, s, a, s_)
            # compute all q values for all heads
            q_values = []
            q_val_base = self.stl_q_net.forward_base(s)
            for head_idx in range(self.num_temporal_ops):
                q_values.append(self.stl_q_net.forward_head(q_val_base, head_idx))
            q_values = torch.stack(q_values)
            targets = self.td_error_vector

            loss_func = torch.nn.SmoothL1Loss()
            loss = loss_func(q_values, targets.clone().detach())


            # backward optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
                self.update_target_network()
                self.reset_td_errors()
                self.iterations_since_last_target_update = 0        
                
def rollout(env, agent, param, i_episode, testing=False, visualize=False):
    # execute an episode of the policy acting in the environment
    state, _ = env.reset()
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    if testing & visualize:
        s = torch.tensor(state['mdp']).type(torch.float)
        print(0, state['mdp'], agent.stl_q_net(s).argmax().to_tensor(0).numpy())
        print(agent.stl_q_net(s))
    for t in range(1, param['T']):  # Don't infinite loop while learning
        action = agent.select_action(state, testing)
        
        next_state, cost, done, info = env.step(action, False)  # check is_eps
        reward = info['is_accepting']
        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            print(t, next_state['mdp'], action)
        if not testing:
            agent.collect(state['mdp'], action, reward, next_state['mdp'],)
            agent.buffer.mark()
        if visualize:
            env.render()
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        if done:
            break
        state = next_state

    return ep_reward, disc_ep_reward, t

def run_Q_STL(param, env):
    agent = STL_Q_learning(env.observation_space, env.action_space, param['gamma'], param)
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
            method = 'Adam'
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