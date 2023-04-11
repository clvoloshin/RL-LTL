import torch
import torch.nn as nn
import mtl
import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from utls.utls import STL_IDS, STLNode, parse_stl_into_tree
from policies.dqn import BufferSTL, DQNSTL

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
        # shared parameters across all heads
        self.stl_q_net = DQNSTL(env_space, act_space, param, self.num_temporal_ops)
        self.stl_q_target = DQNSTL(env_space, act_space, param, self.num_temporal_ops)
        all_params = []
        for head in self.stl_q_net.heads:
            all_params.extend(head.parameters())
        all_params.extend(self.stl_q_net.actor_base.parameters())
        self.optimizer = torch.optim.Adam([
                {'params': all_params, 'lr': param['q_learning']['lr']},
            ])

        self.update_target_network()
        self.rho_alphabet = rho_alphabet
        self.num_rho = len(self.rho_alphabet)
    
        self.buffer = BufferSTL(env_space.shape, self.num_rho, param['replay_buffer_size'])
        self.temp = param['q_learning']['init_temp']
        
        self.n_batches = param['q_learning']['batches_per_update']
        self.batch_size = param['q_learning']['batch_size']
        self.iterations_per_target_update = param['q_learning']['iterations_per_target_update']
        self.iterations_since_last_target_update = 0
        
        self.gamma = gamma
        self.n_mdp_actions = act_space.n
        self.td_error_vector = torch.zeros(self.num_temporal_ops, self.batch_size, self.n_mdp_actions)

    
    def update_target_network(self):
        # copy current_network to target network
        self.stl_q_target.load_state_dict(self.stl_q_net.state_dict())

    def select_action(self, state, is_testing):
        if is_testing or (np.random.uniform() > self.temp):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = self.stl_q_net.act(state_tensor)

            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = self.stl_q_net.random_act(state_tensor)
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
        self.td_error_vector = torch.zeros(self.num_temporal_ops, self.batch_size, self.n_mdp_actions, requires_grad=True)
        
    def collect(self, s, a, rhos, s_):
        self.buffer.add(s, a, rhos, s_)
    
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
                    curr.set_ordering(num_temporal_ops)
                    num_temporal_ops += 1
            for child in curr.children:
                queue.append(child)
        return num_temporal_ops
    
    def recurse_node(self, current_node, s, act, rhos, s_next):
        cid = current_node.id
        if cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = rhos[:, self.rho_alphabet.index(current_node.rho)]
            return phi_val
        elif cid in ["&", "|"]:
            all_phi_vals = []
            for child in current_node.children:
                all_phi_vals.append(self.recurse_node(child, s, act, rhos, s_next))
            # and case and or case are min and max, respectively
            phi_val = torch.min(torch.stack(all_phi_vals), dim=0).values if cid == "&" else torch.max(torch.stack(all_phi_vals), dim=0).values
            return phi_val
        elif cid == "~":  # negation case
            phi_val = self.recurse_node(current_node.children[0], s, act, rhos, s_next)
            return -1 * phi_val
        else:  # G or E case: just get it by recursing with a single child
            phi_val = self.recurse_node(current_node.children[0], s, act, rhos, s_next)
        q_s_next = self.stl_q_net.forward_base(s_next)
        # operator-specific head
        #q_s_next_head = self.stl_q_net.forward_head(s_next, current_node.order)
        q_s_next_head = self.stl_q_net.forward_head(q_s_next, current_node.order)
        q_action = q_s_next_head[torch.arange(q_s_next_head.shape[0]), act]  # TODO: is there a smarter way to do this?
        if cid == "G":
            #import pdb; pdb.set_trace()
            td_val = torch.minimum(phi_val, self.gamma * q_action)
            self.td_error_vector[current_node.order, :,  act] = td_val.float()
        elif cid == "E":
            #import pdb; pdb.set_trace()    
            td_val = torch.maximum(phi_val, self.gamma * q_action)
            self.td_error_vector[current_node.order, :, act] = td_val.float()
        return phi_val
        
    
    def update(self):
        # update based on the STL recursive semantics
        #TODO: check that our recursive traversal does the same
        # as the DFS search in set_heads
        #TODO: finish
        total_loss = 0
        for _ in range(self.n_batches):
            self.iterations_since_last_target_update += 1
            with torch.no_grad():
                s, a, rhos, s_, = self.buffer.sample(self.batch_size)
                s = torch.tensor(s).type(torch.float)
                s_ = torch.tensor(s_).type(torch.float)
                rhos = torch.tensor(rhos)
                a = torch.tensor(a)
                # compute TD values
                self.reset_td_errors()
                self.recurse_node(self.stl_tree, s, a, rhos, s_)
            # compute all q values for all heads
            q_values = []  ##  TODO: check if this is ok
            q_val_base = self.stl_q_net.forward_base(s)
            for head_idx in range(self.num_temporal_ops):
                #q_values.append(self.stl_q_net.forward_head(s, head_idx))
                q_values.append(self.stl_q_net.forward_head(q_val_base, head_idx))
            q_values = torch.stack(q_values)
            targets = self.td_error_vector
            #import pdb; pdb.set_trace()
            loss_func = torch.nn.SmoothL1Loss()
            loss = loss_func(q_values, targets.clone().detach())
            total_loss += loss

            # backward optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.iterations_since_last_target_update % self.iterations_per_target_update) == 0:
                self.update_target_network()
                #self.reset_td_errors()
                self.iterations_since_last_target_update = 0
        return total_loss / self.n_batches        
                
def rollout(env, agent, param, i_episode, testing=False, visualize=False):
    # execute an episode of the policy acting in the environment
    states = []
    state, _ = env.reset()
    states.append(state)
    s = torch.tensor(state).type(torch.float)
    max_q_val = torch.max(agent.stl_q_net(s)).detach().numpy()
    if not testing: agent.buffer.restart_traj()
    if testing & visualize:
        s = torch.tensor(state).type(torch.float)
        print(0, state, agent.stl_q_net(s).argmax().numpy())
        print(agent.stl_q_net(s))
    for t in range(1, param['q_learning']['T']):  # Don't infinite loop while learning
        action = agent.select_action(state, testing)
        
        next_state, cost, done, info = env.step(action)  # check is_eps
        rhos = info['rho']
        
        if testing & visualize:
            s = torch.tensor(next_state).type(torch.float)
            print(t, next_state, action)
        if not testing:
            agent.collect(state, action, rhos, next_state,)
            agent.buffer.mark()

        if done:
            break
        state = next_state
        states.append(next_state)

    all_rho_vals = env.episode_rhos
    if visualize:
        env.render(states=states, save_dir=logger.get_dir() + '/' + "episode_" + str(i_episode))
    return all_rho_vals, max_q_val, t

def run_Q_STL(param, env):
    stl_tree = parse_stl_into_tree(param['ltl']['formula'])
    varphi = mtl.parse(param['ltl']['formula'])
    agent = STL_Q_learning(stl_tree, env.observation_space, env.action_space, env.rho_alphabet, param['gamma'], param)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()
    current_loss = 0  # placeholder value
    all_losses = [current_loss]

    for i_episode in range(param['q_learning']['n_traj']):
        # TRAINING
        all_rhos, max_q_val, t = rollout(env, agent, param, i_episode, testing=False)
        stl_val = varphi(all_rhos)
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update().item()
            all_losses.append(current_loss)
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                all_rhos, max_q_val, t = rollout(env, agent, param, test_iter, testing=True, visualize= ((i_episode % 50) == 0) & (test_iter == 0) )
                stlval = varphi(all_rhos)
                test_data.append((stlval, t))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [stl_val]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            #import pdb; pdb.set_trace()
            success_history += [test_data[:, 0].mean()]
            method = 'STLQ'
            plot_something_live(axes, [np.arange(len(history)),  np.arange(len(all_losses))], [history, all_losses], method)
            logger.logkv('Iteration', i_episode)
            logger.logkv('Method', method)
            logger.logkv('Success', success_history[-1])
            logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            logger.logkv('EpisodeRhos', stl_val)
            logger.logkv('ExpectedQVal', max_q_val)  
            logger.logkv('LossVal', current_loss)
            logger.logkv('TimestepsAlive', avg_timesteps)
            logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['q_learning']['T'])
            logger.logkv('ActionTemp', agent.temp)
            
            logger.dumpkvs()
            
    plt.close()