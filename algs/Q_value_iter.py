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

def BO(x, eps=0.):
    # Boltzmann Operator
    # https://en.wikipedia.org/wiki/Smooth_maximum#See_also
    x = np.array(x)
    return np.sum(x * np.exp(eps * x)) / np.sum(np.exp(eps * x))


def softplus(x, eps=1.):
    #softmax/softmin
    return eps * scipy.special.logsumexp(np.array(x)/eps)

def softminus(x, eps=1.):
    return -softplus(-np.array(x))


class Q_actual:
    def __init__(self, tree, rho_alphabet, n_states, env_space, act_space, gamma, param) -> None:
        self.stl_tree = tree
        self.rho_alphabet = rho_alphabet
        self.outer_most_neg = True if self.stl_tree.id in ["~", "!"] else False
        self.num_temporal_ops, self.ordering = self.set_ordering()
        self.Q = ma.zeros((self.num_temporal_ops, n_states, env_space['buchi'].n, act_space['total']))
        self.N = ma.zeros((self.num_temporal_ops, n_states, env_space['buchi'].n, act_space['total']))
        
        self.alpha = 1.

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
            self.Q[..., buchi, eps:] = ma.masked
            self.N[..., buchi, eps:] = ma.masked
        
        self.gamma = gamma
        self.n_mdp_actions = act_space['mdp'].n
        self.mapping = {}
        self.states_to_idx = {}
        self.idx_to_states = {}
    
    def value_iteration(self, env):  
        if len(self.idx_to_states) == 0:
            output, _ = env.reset()
            self.states_to_idx[tuple(output['mdp'])] = 0
            self.idx_to_states[0] = output['mdp']

        iter = 0
        while iter < 1000:
            iter += 1
            eps = 0
            for s in range(self.Q.shape[1]):
                for b in range(self.Q.shape[2]):
                    for a in range(self.Q.shape[3]):
                        try: 
                            if self.Q[0, s, b, a].mask: 
                                continue
                        except:
                            pass

                        if (s, b, a) not in self.mapping:
                            next_state, cost, done, info = env.simulate_step(self.idx_to_states[s], b, a,  a >= self.n_mdp_actions)

                            s_ = np.round(next_state['mdp'], 3)
                            b_ = next_state['buchi']
                            r = info['rho']

                            if tuple(s_) not in self.states_to_idx:
                                idx_ = len(self.idx_to_states)
                                self.states_to_idx[tuple(s_)] = idx_
                                self.idx_to_states[idx_] = s_
                            
                            s__idx = self.states_to_idx[tuple(s_)]

                            self.mapping[(s, b, a)] = (s__idx, b_, r)
                        
                        (s_, b_, rhos) = self.mapping[(s, b, a)]
                        Qs = self.Q[0, s_, b_, :]
                        max_actions = Qs.argmin() if self.outer_most_neg else Qs.argmax()
                        # max_actions = np.random.choice(np.flatnonzero(Qs == Qs.max()))
                        
                        self.reset_td_errors()
                        self.recurse_node(self.stl_tree, s, b, max_actions, rhos, s_, b_)
                        for head in range(len(self.td_error_vector)):
                            eps += np.abs(self.Q[head, s, b, a] - self.td_error_vector[head])
                            self.Q[head, s, b, a] = (1-self.alpha) * self.Q[head, s, b, a] + self.alpha * self.td_error_vector[head]
            
            if eps < 1e-3:
                break
            print(iter, eps, self.Q[0, 0, 0, :].argmax())
            # print(iter, eps)
        
        # self.rollout(30)
        print(f'Final eps: {eps}')
        import pdb; pdb.set_trace()
        
    def reset_td_errors(self):
        self.td_error_vector = np.zeros((self.num_temporal_ops))

    def recurse_node(self, current_node, s, b, act, rhos, s_next, b_next):
        cid = current_node.id
        if cid == "rho":
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
    s, b = 0, 0
    states = [agent.idx_to_states[s]]
    for _  in range(T):

        Qs = agent.Q[0, s, b, :]
        # agent.outer_most_neg = True
        a = Qs.argmin() if agent.outer_most_neg else Qs.argmax()
        s_, b_, rhos = agent.mapping[(s, b, a)]
        print(s, b, a, s_, b_)
        print(Qs)
        print(agent.idx_to_states[s], agent.idx_to_states[s_])
        states.append(agent.idx_to_states[s_])
        s = s_
        b = b_
    
    img = env.render(states=states, save_dir=None)
    PIL.Image.fromarray(img).show()
    runner.log({"training": wandb.Image(img)})

def run_value_iter(param, runner, env, second_order = False):
    
    stl_tree = parse_stl_into_tree(param['ltl']['formula'])
    agent = Q_actual(stl_tree, env.mdp.rho_alphabet, env.mdp.n_implied_states, env.observation_space, env.action_space, param['gamma'], param)
    agent.value_iteration(env)
    rollout(agent, env, runner, 30)