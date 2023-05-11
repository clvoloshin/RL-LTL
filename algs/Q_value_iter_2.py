import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from utls.utls import STL_IDS, STLNode, parse_stl_into_tree

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
import numpy.ma as ma
import pandas as pd
import wandb
import scipy
import PIL
import networkx as nx

class Q_actual:
    def __init__(self, reward_funcs, tree, rho_alphabet, n_states, env_space, act_space, gamma, accepting_states, param) -> None:
        self.stl_tree = tree
        self.reward_funcs = reward_funcs
        self.lambda_penalties = {bstate: 10 for bstate in range(env_space['buchi'].n)}
        self.lambda_multiplier = 1.5  # TODO: figure out how to set this hyperparam?
        self.rho_alphabet = rho_alphabet
        self.max_lambda_updates = 5  # TODO: find a better way of evaluating empirical satisfaction
        self.outer_most_neg = True if self.stl_tree.id in ["~", "!"] else False
        self.num_temporal_ops, self.ordering = self.set_ordering()
        self.Q = ma.zeros((n_states, env_space['buchi'].n, act_space['total']))
        self.N = ma.zeros((self.num_temporal_ops, n_states, env_space['buchi'].n, act_space['total']))
        self.outer_target = ma.zeros((n_states, env_space['buchi'].n, act_space['total']))
        self.alpha = 1.
        self.accepting_states = accepting_states

        # Instantiate bias
        # for head in range(self.num_temporal_ops):
            # operator = self.ordering[head]
            # if operator not in ["F", "E", "G"]: continue
            # self.Q[head,...] = -1 if operator in ["F", "E"] else 1
        # self.Q += 1.

        # Mask actions not available
        for buchi in range(env_space['buchi'].n):
            try:
                eps = act_space['total'] - 1 + act_space[buchi].n
            except:
                eps = act_space['total'] - 1
            self.Q[:, buchi, eps:] = ma.masked
            self.N[..., buchi, eps:] = ma.masked
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
        self.mapping = {}
        self.states_to_idx = {}
        self.idx_to_states = {}
    
    def ltl_reward_1(self, env, rhos, edge, terminal, b, b_):
        if terminal: #took sink
            return 0, True
        if b_ in self.accepting_states:
            return 1, False
        return 0, False
    
    def ltl_reward_3(self, env, rhos, edge, terminal, b, b_):
        # # TODO: do this better??
        if terminal: #took sink
            return -1, True
        
        if b in self.reward_funcs:
            reward_func = self.reward_funcs[b][0]
            r = self.recurse_node(reward_func.stl, None, None, None, rhos, None, None)
            return r, False
        else: # epsilon transition
            return 0, False
    
    def constrained_reward(self, env, rhos, edge, terminal, b, b_, mdp_reward):
        # we can replace the ltl reward call with whatever reward structure we want
        ltl_reward, done = self.ltl_reward_1(env, rhos, edge, terminal, b, b_)
        edge_lambda = self.lambda_penalties[b]
            #return ltl_reward, done
        return mdp_reward + edge_lambda * ltl_reward, done
        
    def update_lambdas(self):
        for b in self.lambda_penalties:
            self.lambda_penalties[b] *= self.lambda_multiplier
            self.outer_target[:, b, :] = self.Q[:, b, :]

    def constrained_optimization(self, env):
        # outer loop
        # termination conditions: we satisfy the spec, and there's no change in the policy
        itr = 0
        while itr < self.max_lambda_updates:
            print("=" * 10, " OUTER LOOP ITER {}".format(itr), "=" * 10)
            self.value_iteration(env)
            emp_sat = self.empirical_satisfaction(env)
            self.update_lambdas()
            if emp_sat: # we are satisfying the spec so done
                break
            itr += 1
        
    def single_rollout(self, env):
        output, _ = env.reset()
        s, b = self.states_to_idx[tuple(output['mdp'])], output['buchi']
        states = [self.idx_to_states[s]]
        num_accepting_visits = 0
        for _  in range(30):

            Qs = self.Q[s, b, :]
            # agent.outer_most_neg = True
            #import pdb; pdb.set_trace()
            a = Qs.argmin() if self.outer_most_neg else Qs.argmax()
            s_, b_, rew, rhos, edge, terminal = self.mapping[(s, b, a)]
            states.append(self.idx_to_states[s_])
            s = s_
            b = b_
            if b in env.automaton.accepting_sets:
                num_accepting_visits += 1
        return num_accepting_visits, b
    
    def empirical_satisfaction(self, env, num_trajectories=1, min_num_accept_visits=1):
        # determine if the trajectory has at least 'empirically' satisfied spec
        sat = True
        for _ in range(num_trajectories):
            num_accept_visits, curr_b = self.single_rollout(env)
            if num_accept_visits < min_num_accept_visits:
                sat = False
                break
            if curr_b not in self.reward_funcs:
                sat = False
                break
        return sat
                        
    def value_iteration(self, env):  
        if len(self.idx_to_states) == 0:
            output, _ = env.reset()
            self.states_to_idx[tuple(output['mdp'])] = 0
            self.idx_to_states[0] = output['mdp']

        iter = 0
        while iter < 1000:
            iter += 1
            eps = 0
            for s in range(self.Q.shape[0]):
                for b in range(self.Q.shape[1]):
                    for a in range(self.Q.shape[2]):
                        try: 
                            if self.Q[s, b, a].mask: 
                                continue
                        except:
                            pass

                        if (s, b, a) not in self.mapping:
                            next_state, reward, done, info = env.simulate_step(self.idx_to_states[s], b, a,  a >= self.n_mdp_actions)

                            s_ = np.round(next_state['mdp'], 3)
                            b_ = next_state['buchi']
                            r = info['rho']
                            edge = info['edge']
                            terminal = info['is_rejecting']

                            if tuple(s_) not in self.states_to_idx:
                                idx_ = len(self.idx_to_states)
                                self.states_to_idx[tuple(s_)] = idx_
                                self.idx_to_states[idx_] = s_
                            
                            s__idx = self.states_to_idx[tuple(s_)]

                            self.mapping[(s, b, a)] = (s__idx, b_, reward, r, edge, terminal)
                        
                        (s_, b_, rew, rhos, edge, terminal) = self.mapping[(s, b, a)]

                        #r, done = self.reward(env, rhos, edge, terminal, b, b_)
                        r, done = self.constrained_reward(env, rhos, edge, terminal, b, b_, rew)
                        # if r > 0: import pdb; pdb.set_trace()
                        target = (r + self.gamma * self.Q[s_, b_, :].max()) * (1-done) + done * r/(1-self.gamma)
                        eps += np.abs(self.Q[s, b, a] - target)
                        self.Q[s, b, a] = self.Q[s, b, a] * (1-self.alpha) + self.alpha * target

            if eps < 1e-3:
                break
            print(iter, eps, self.Q[0, 0, :].argmax())
            # print(iter, eps)
        
        # self.rollout(30)
        print(f'Final eps: {eps}')
        #import pdb; pdb.set_trace()
        
    def reset_td_errors(self):
        self.td_error_vector = np.zeros((self.num_temporal_ops))

    def recurse_node(self, current_node, s, b, act, rhos, s_next, b_next):
        cid = current_node.id
        if cid == 'True':
            return 1
        elif cid == 'False':
            return -1
        elif cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = rhos[self.rho_alphabet.index(current_node.rho)]
            return phi_val
        elif cid in ["&", "|"]:
            all_phi_vals = []
            for child in current_node.children:
                all_phi_vals.append(self.recurse_node(child, s, b, act, rhos, s_next, b_next))
            # and case and or case are min and max, respectively
            # if min(all_phi_vals) != all_phi_vals[0]:
            # if s == 0:
            #     if act != 0:
            #         import pdb; pdb.set_trace()
            phi_val = min(all_phi_vals) if cid == "&" else max(all_phi_vals)
            # phi_val = BO(all_phi_vals, -10.) if cid == "&" else BO(all_phi_vals, 10.)
            return phi_val
        elif cid in ["~", "!"]:  # negation case
            phi_val = self.recurse_node(current_node.children[0], s, b, act, rhos, s_next, b_next)
            return -1 * phi_val
        else:  # G or E or X case: just get it by recursing with a single child
            phi_val = self.recurse_node(current_node.children[0], s, b, act, rhos, s_next, b_next)
            # td_val = phi_val
        
        # q_action = q_s_next_head[torch.arange(q_s_next_head.shape[0]), act]  # TODO: is there a smarter way to do this?

        ## originally: Q(s) ~=   min(r, gamma * max_{a'} Q(s', a')) 
        ## ours:       Q(s) ~=       r + gamma * max_{a'} Q(s', a')
        q_action = self.Q[current_node.order, s_next, b_next, act]
        
        if cid == "G":
            phi_val = min(phi_val, self.gamma * q_action) #BO([phi_val, self.gamma * q_action], -100.) #min
            self.td_error_vector[current_node.order] = phi_val
        elif cid in ["E", "F"]:
            phi_val = max(phi_val, self.gamma * q_action) #BO([phi_val, self.gamma * q_action], 100.) #max
            self.td_error_vector[current_node.order] = phi_val
        elif cid == "X":
            phi_val = self.gamma * q_action
            self.td_error_vector[current_node.order] = phi_val
        
        return phi_val
    
    def set_ordering(self):
        num_expr = 0
        queue = [self.stl_tree]
        num_temporal_ops = 0
        order = []
        while len(queue) > 0:
            curr = queue.pop(0)
            if curr.id != "rho":
                num_expr += 1
                # set the head that'll correspond to this operator
                if curr.id in ["G", "E", "F", "X"]:
                    curr.set_ordering(num_temporal_ops)
                    order.append(curr.id)
                    num_temporal_ops += 1
            for child in curr.children:
                queue.append(child)
        return num_temporal_ops, order

def rollout(agent, env, runner, T):
    output, _ = env.reset()
    s, b = agent.states_to_idx[tuple(output['mdp'])], output['buchi']
    states = [agent.idx_to_states[s]]
    for _  in range(T):

        Qs = agent.Q[s, b, :]
        # agent.outer_most_neg = True
        #import pdb; pdb.set_trace()
        a = Qs.argmin() if agent.outer_most_neg else Qs.argmax()
        s_, b_, rew, rhos, edge, terminal = agent.mapping[(s, b, a)]
        print(s, b, a, s_, b_)
        print(Qs)
        print(agent.idx_to_states[s], agent.idx_to_states[s_])
        states.append(agent.idx_to_states[s_])
        s = s_
        b = b_
    node_colors = ["green" if node in env.automaton.accepting_sets else "lightblue" for node in range(env.automaton.n_states)]
    #import pdb; pdb.set_trace()
    buchi_graph = env.automaton.automaton.G
    #edge_labels = {edg: buchi_graph.edges[edg]['condition'] for edg in buchi_graph.edges}

    #nx.draw_networkx_edge_labels(buchi_graph, pos=nx.spring_layout(buchi_graph), edge_labels=edge_labels, font_color='red', font_size=3)

    nx.draw_networkx(buchi_graph, pos=nx.spring_layout(buchi_graph), node_color=node_colors)
    import pdb; pdb.set_trace()
    plt.savefig("automaton.png")
    img = env.render(states=states, save_dir=None)
    PIL.Image.fromarray(img).show()
    runner.log({"training": wandb.Image(img)})
    return s, b # hacky

def run_value_iter(param, runner, env, second_order = False):
    
    
    
    for path in sorted(nx.simple_cycles(env.automaton.automaton.G)):
        print(path)
    
    ## G(F(g) & ~b & ~r & ~y)
    #reward_funcs = {0: [env.automaton.edges(0, 1)[0], env.automaton.edges(0, 0)[0]], 1: [env.automaton.edges(1, 0)[0]]}
    
    ## G(F(y & X(F(r)))) & G~b
    reward_funcs = {0: [env.automaton.edges(0, 1)[0]], 1: [env.automaton.edges(1, 2)[0]], 2: [env.automaton.edges(2, 0)[0]]}
    #import pdb; pdb.set_trace()
    ## F(G(y))
    #reward_funcs = {1: [env.automaton.edges(1, 1)[0]]}
    
    ## F(r & XF(G(y)))
    #reward_funcs = {2: [env.automaton.edges(2, 2)[0]]}  
    #import pdb; pdb.set_trace()
    ## F(r & XF(g & XF(y))) & G~b
    #reward_funcs = {2: [env.automaton.edges(2, 3)[0]], 3: [env.automaton.edges(3, 1)[0]], 1: [env.automaton.edges(1, 0)[0]], 0: [env.automaton.edges(0, 0)[0]]}  
    # reward_funcs = {0: [env.automaton.edges(0, 0)[0]]}

    

    # print(reward_funcs)
    stl_tree = parse_stl_into_tree(param['ltl']['formula'])
    agent = Q_actual(reward_funcs, stl_tree, env.mdp.rho_alphabet, env.mdp.n_implied_states, env.observation_space, env.action_space, param['gamma'],
                     env.automaton.accepting_sets, param)
    #agent.value_iteration(env)
    agent.constrained_optimization(env)
    rollout(agent, env, runner, 30)