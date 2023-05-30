import numpy as np
import torch


class ConstrainedOptimization():
    
    def __init__(self, param, runner, env) -> None:
        self.lamb = param['lambda']
        self.lambda_multiplier = param["mutiplier"]
        self.max_update_iters = param["num_outer_iters"]
        self.accepting_states = torch.tensor(env.automaton.accepting_sets)
        self.terminal_states = torch.tensor([state for state in env.automaton.states if state.terminal()])
        self.param = param
        self.runner = runner
        

    def ltl_reward_1(self, rhos, terminal, b, b_):
        return torch.isin(b_, self.accepting_states).float(), torch.isin(b_, self.terminal_states)
    
    def ltl_reward_3(self, rhos, terminal, b, b_):
        # # TODO: change this to handle tensor inputs 
        if terminal: #took sink
            return -1, True
        
        if b in self.reward_funcs:
            reward_func = self.reward_funcs[b][0]
            r = self.evaluate_buchi_edge(reward_func.stl, rhos)
            return r, False
        else: # epsilon transition
            return 0, False
    
    def constrained_reward(self, rhos, terminal, b, b_, mdp_reward):
        # we can replace the ltl reward call with whatever reward structure we want
        ltl_reward, done = self.ltl_reward_1(rhos, terminal, b, b_)
            #return ltl_reward, done
        return mdp_reward + self.lamb * ltl_reward.squeeze(), done
        
    def update_lambdas(self):
        self.lamb *= self.lambda_multiplier

    def constrained_optimization(self, env, policy_alg):
        # outer loop
        # termination conditions: we satisfy the spec, and there's no change in the policy
        itr = 0
        while itr < self.max_update_iters:
            print("=" * 10, " OUTER LOOP ITER {}".format(itr), "=" * 10)
            policy_alg(self.param, env, self.constrained_reward)
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
    
    def evaluate_buchi_edge(self, ast_node, rhos):
        cid = ast_node.id
        if cid == 'True':
            return 1
        elif cid == 'False':
            return -1
        elif cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = rhos[self.rho_alphabet.index(ast_node.rho)]
            return phi_val
        elif cid in ["&", "|"]:
            all_phi_vals = []
            for child in ast_node.children:
                all_phi_vals.append(self.evaluate_buchi_edge(child, rhos))
            # and case and or case are min and max, respectively
            phi_val = min(all_phi_vals) if cid == "&" else max(all_phi_vals)
            return phi_val
        elif cid in ["~", "!"]:  # negation case
            phi_val = self.evaluate_buchi_edge(ast_node.children[0], rhos)
            return -1 * phi_val