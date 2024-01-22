from abc import ABCMeta, abstractmethod
import gym
#import gymnasium.spaces
import numpy as np
import torch
from copy import deepcopy
import time

def timeit(f):
    def wrapped(*args, **kw):
        tic = time.time()
        result = f(*args, **kw)
        toc = time.time()
        print('{} {} {} ms'.format('Sim', f.__name__, (toc - tic) * 1000))
        return result
    
    return wrapped

class AbstractEnv(metaclass=ABCMeta):
    @abstractmethod
    def label(state):
        return
    
    @abstractmethod
    def reset(self):
        return
    
    @abstractmethod
    def step(self, action):
        return
    
    def from_sparse_to_full(self, D):
        T = np.zeros((self.observation_space.n, self.action_space.n+1, self.observation_space.n))
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n + 1):
                for next_state in range(self.observation_space.n):
                    try:
                        T[state][action][next_state] = self.T[state][action][next_state]
                    except:
                        pass
                
                    # if action == self.action_space.n:
                    #     T[state][action][-1] = 1
        
        C = []
        for state in range(self.observation_space.n):
            C.append([0]*(self.action_space.n + 1))
            for action in range(self.action_space.n + 1):
                try:
                    C[state][action] = self.cost[state][action]
                except:
                    C[state][action] = D

        return T, C

class Simulator(gym.Env):
    def __init__(self, mdp, automaton, lambda_val, qs_lambda=1.0, reward_type=2):
        self.mdp = mdp
        self.automaton = automaton
        spaces = {
            'mdp': self.mdp.observation_space,
            'buchi': gym.spaces.Discrete(self.automaton.n_states)
            }
        self.observation_space = spaces #self.mdp.observation_space.shape[0] + self.automaton.n_states

        eps_per_state = [len(val) for key,val in self.automaton.automaton.eps.items()] + [0]
       
        spaces = {
            'mdp': self.mdp.action_space,
        }
        spaces.update({key: gym.spaces.Discrete(len(val)) for key, val in self.automaton.automaton.eps.items()})
        self.action_space = dict(spaces)
        
        try:
            self.action_space['total'] = sum([val.n for key, val in self.action_space.items()])
        except:
            self.action_space['total'] = 1 + max(eps_per_state)
        
        self.accepting_states = set()
        self.rejecting_states = set()
        self.inf_often = []
        self.lambda_val = lambda_val
        self.qs_lambda_val = qs_lambda
        self.reward_type = reward_type
        all_accepting_cycles = []
        # import pdb; pdb.set_trace()
        print("finding cycles...")
        for state in self.automaton.automaton.accepting_states:
            cycles = self.find_min_accepting_cycles(state)
            all_accepting_cycles.extend(cycles)
        self.all_accepting_cycles = all_accepting_cycles
        self.acc_cycle_edge_counts = self.get_rewarding_edge_counts()
        self.fixed_cycle = None
        self.num_cycles = len(self.all_accepting_cycles)
        self.previous_rhos = np.zeros(self.num_cycles)  # initialize this to nothing for now
        self.qs_resets = [False] * self.num_cycles
        print("Found {} cycles".format(self.num_cycles))
        if self.reward_type % 2 != 0: ## IF we have a fixed reward:
            self.num_cycles = 1 # only reward one thing
            self.acc_cycle_edge_counts = [1.]
        # import pdb; pdb.set_trace()
    
    def get_rewarding_edge_counts(self):
        return [len(cyc) for cyc in self.all_accepting_cycles]
            
    def unnormalize(self, states):
        try:
            return self.mdp.unnormalize(states)
        except:
            return states
    
    def did_succeed(self, x, a, r, x_, d, t):
        return self.mdp.did_succeed(x['mdp'], a, r, x_['mdp'], d, t)
        # return self.mdp.did_succeed(x[:self.mdp.observation_space.shape[0]], a, r, x_[:self.mdp.observation_space.shape[0]], d, t)
    
    def label(self, state):
        raise NotImplemented
        # return self.mdp.label(state[:self.mdp.observation_space.shape[0]])        
             
    def reset(self, make_aut_init_state_random=False, *args, **kwargs):
        try:
            #allow reset at any point, even if using Monitor
            self.mdp.stats_recorder.done = True
        except:
            pass

        wrapped_state, _ = self.mdp.reset()
        if isinstance(wrapped_state, dict):
            state = wrapped_state['state']
            label_state = wrapped_state['data']
            label, _ = self.mdp.label(wrapped_state)
        else:
            state = wrapped_state
            label, _ = self.mdp.label(state)
        
        self.automaton.reset()
        if make_aut_init_state_random:
            self.automaton.set_state(np.random.choice(self.automaton.n_states - 1))
        automaton_state, edge = self.automaton.step(label)

        return {'mdp': wrapped_state, 'buchi': automaton_state}, {'edge': edge}

        # one_hot = np.zeros(self.automaton.n_states)
        # one_hot[automaton_state] = 1.
        # return np.hstack([state, one_hot]), {}
    def decay_lambda(self, decay_rate, min_lambda, decay_type):
    
        if decay_type == 'linear':
            self.lambda_val = self.lambda_val - decay_rate
        elif decay_type == 'exponential':
            self.lambda_val = self.lambda_val * decay_rate
        else:
            raise NotImplemented
        
        if (self.lambda_val <= min_lambda):
            self.lambda_val = min_lambda
        
        self.lambda_val = round(self.lambda_val, 4)
        return self.lambda_val
        #print(f'Setting temperature: {self.temp}')

    def next_buchi(self, mdp_state, desired_current_aut_state, eps_action=None):
        # The mdp_state is for getting the label.
        if eps_action is None:
            current_aut_state = self.automaton.get_state()
            label, _ = self.mdp.label(mdp_state)
            self.automaton.set_state(desired_current_aut_state)
            automaton_state, edge = self.automaton.step(label)
            self.automaton.set_state(current_aut_state)
            # if (automaton_state in self.automaton.automaton.accepting_states) or (automaton_state != current_aut_state):
            if (automaton_state in self.automaton.automaton.accepting_states):
                accepting_rejecting_neutral = 1
            elif self.automaton.is_rejecting(automaton_state):
                accepting_rejecting_neutral = -1
            else:
                accepting_rejecting_neutral = 0
            return automaton_state, accepting_rejecting_neutral
        else:
            current_aut_state = self.automaton.get_state()
            label, _ = self.mdp.label(mdp_state)
            self.automaton.set_state(desired_current_aut_state)
            automaton_state, edge = self.automaton.epsilon_step(eps_action)
            automaton_state, edge = self.automaton.step(label) # Do we take this step right now???
            self.automaton.set_state(current_aut_state)
            # if (automaton_state in self.automaton.automaton.accepting_states) or (automaton_state != current_aut_state):
            if (automaton_state in self.automaton.automaton.accepting_states):
                accepting_rejecting_neutral = 1
            elif self.automaton.is_rejecting(automaton_state):
                accepting_rejecting_neutral = -1
            else:
                accepting_rejecting_neutral = 0
            return automaton_state, accepting_rejecting_neutral
    
    def ltl_reward_1_scalar(self, terminal, b, b_):
        if terminal: #took sink
            return 0, True
            #return -1, True
        if b_ in self.automaton.automaton.accepting_states:
            return 1, False
        return 0, False

    def ltl_reward_1(self, terminal, b, b_):
        # print(f"b_ shape: {b_.shape}")
        # print(f"accepting states: {self.accepting_states}")
        reward, terminal = self.ltl_reward_1_scalar(terminal, b, b_)
        return np.array([reward]) * self.lambda_val, terminal
        # return torch.isin(b_, self.accepting_states).float(), terminal

    def ltl_reward_2(self, terminal, b, b_):
        cycle_rewards = []

            #return -1, True
        for idx, buchi_cycle in enumerate(self.all_accepting_cycles):
            # if terminal: # terminal state
            #     cycle_rewards.append(-1.0)
                        # if we take a sink transition, penalize that.
            # if (not self.automaton.is_rejecting(b)) and self.automaton.is_rejecting(b_):
            #     # if we take a sink transition
            #     cycle_rewards.append(-1.0 * self.lambda_val)
            if b in buchi_cycle:
                # if b in self.automaton.automaton.accepting_states and b_ not in self.automaton.automaton.accepting_states: 
                #     cycle_rewards.append(0.0) # if we're leaving an accept state, don't reward it
                if b_ == buchi_cycle[b].child.id:
                    cycle_rewards.append(1.0 * self.lambda_val  / self.acc_cycle_edge_counts[idx])
                    # else:
                    #     cycle_rewards.append(0.0)
                else:
                    cycle_rewards.append(0.0)
            else: # epsilon transition or non-cycle transition
                cycle_rewards.append(0.0)
        if terminal: #took sink
                return np.array(cycle_rewards), True
        return np.array(cycle_rewards), False
    
    def ltl_reward_3(self, terminal, b, b_):
        if terminal:
            reward = 0
        if b in self.fixed_cycle:
            if b_ == self.fixed_cycle[b].child.id:
                reward = 1.0 * self.lambda_val 
            else:
                reward = 0.0
        return np.array([reward]), not terminal

    def ltl_reward_zero(self, terminal, b, b_, rhos):
        # check if in a cycle, then evaluate the quantitative semantics
        cycle_rewards = []
        for idx, buchi_cycle in enumerate(self.all_accepting_cycles):
            # if we take a sink transition, penalize that.
            if (not self.automaton.is_rejecting(b)) and self.automaton.is_rejecting(b_):
                # if we take a sink transition
                min_delta = self.mdp.rho_min - self.mdp.rho_max
                cycle_rewards.append(min_delta) #* self.qs_lambda_val)
            elif b in buchi_cycle:
                rho = self.evaluate_buchi_edge(buchi_cycle[b].stl, rhos) + 1e-8 # add a small value to avoid the zero xform
                delta = rho - self.previous_rhos[idx]
                delta = delta * self.qs_lambda_val
                # self.previous_rhos[idx] = rho
                if b_ == buchi_cycle[b].child.id:
                    # import pdb; pdb.set_trace()
                    delta = delta + (1.0 * self.lambda_val  / self.acc_cycle_edge_counts[idx])
                    cycle_rewards.append(delta)
                    new_rho = self.evaluate_buchi_edge(buchi_cycle[b_].stl, rhos)
                    self.previous_rhos[idx] = new_rho
                else:
                    cycle_rewards.append(delta)
                    self.previous_rhos[idx] = rho
            else:
                # if we're outside the cycle
                #cycle_rewards.append(min_delta * self.qs_lambda_val) # ignore these
                cycle_rewards.append(0.0)
                # cycle_rewards.append(0.0)

        if terminal:
            return np.array(cycle_rewards), True
        # import pdb; pdb.set_trace()
        return np.array(cycle_rewards), False
    
    def evaluate_buchi_edge(self, stl_node, rhos):
        cid = stl_node.id
        if cid == 'True':
            return 1
        elif cid == 'False':
            return -1
        elif cid == "rho":
            # evaluate the robustness function using the rho belonging to that node
            phi_val = rhos[self.mdp.rho_alphabet.index(stl_node.rho)]
            return phi_val
        elif cid in ["&", "|"]:
            all_phi_vals = []
            for child in stl_node.children:
                all_phi_vals.append(self.evaluate_buchi_edge(child, rhos))
            # and case and or case are min and max, respectively
            phi_val = min(all_phi_vals) if cid == "&" else max(all_phi_vals)
            return phi_val
        elif cid in ["~", "!"]:  # negation case
            phi_val = self.evaluate_buchi_edge(stl_node.children[0], rhos)
            return -1 * phi_val

    def constrained_reward(self, 
                            terminal, 
                            b, 
                            b_, 
                            mdp_reward,
                            rhos
                            ):
        # will have multiple choices of reward structure
        # TODO: add an automatic structure selection mechanism
        if self.reward_type == 2:  # if it's a cycler-based method
            ltl_reward, done = self.ltl_reward_2(terminal, b, b_)
        elif self.reward_type == 0: # use quantitative semantics
            ltl_reward, done = self.ltl_reward_zero(terminal, b, b_, rhos)
        else:
            ltl_reward, done = self.ltl_reward_1(terminal, b, b_) 
        # Here, we moved the lambda calculations to each reward function itself.
        if self.reward_type > 2:
            return ltl_reward, done, {"ltl_reward": ltl_reward, "mdp_reward": mdp_reward}
        return mdp_reward + ltl_reward, done, {"ltl_reward": ltl_reward, "mdp_reward": mdp_reward}
    
    # @timeit
    def step(self, action, is_eps=False):
        current_mdp_state, current_aut_state = self.mdp.get_state(), self.automaton.get_state()

        if is_eps: 
            #epsilon transition
            state = self.mdp.get_state()
            if isinstance(state, dict):
                wrapped_state = state
                state = state['state']
                label, _ = self.mdp.label(wrapped_state)
            else:
                label, _ = self.mdp.label(state)
            if self.mdp.continuous_actions == False:
                automaton_state, edge = self.automaton.epsilon_step(action - self.mdp.action_space.n) # discrete
            else:
                automaton_state, edge = self.automaton.epsilon_step(action - 1) # continuous

            automaton_state, edge = self.automaton.step(label) # Do we take this step right now???
            cost = 0
            done = False
            info = self.mdp.get_info()
            info.update({'edge': edge})

        else:
            output = self.mdp.step(action)
            try:
                state, cost, done, _, info = output 
            except:
                state, cost, done, info = output
            if isinstance(state, dict):
                # label_state = state['data']
                # wrapped_state = state
                # unwrapped_state = state['state']
                label, _ = self.mdp.label(state)
            else:
                label, _ = self.mdp.label(state)
            automaton_state, edge = self.automaton.step(label)
        

        new_info = {'edge': edge, 'prev_mdp_state': current_mdp_state, 'prev_aut_state': current_aut_state , 's_': state, 'aut_state': automaton_state, 'label': label, 'is_accepting': automaton_state in self.automaton.automaton.accepting_states, 'is_rejecting': self.automaton.is_rejecting(automaton_state)}
        try:
            new_info.update(info)
        except:
            import pdb; pdb.set_trace()

        if automaton_state is None:
            import pdb; pdb.set_trace()

        # if automaton_state in self.automaton.automaton.accepting_states: # accepting state
        #     self.accepting_states.add(state)
        # elif automaton_state == (self.automaton.automaton.n_states - 1): # rejecting state
        #     self.rejecting_states.add(state)

        # info.update({'fail': automaton_state in self.rejecting_states, 'goal': automaton_state in self.accepting_states})
        
        # if simulate:
        #     self.mdp.set_state(current_mdp_state)
        #     self.automaton.set_state(current_aut_state)

        return {'mdp': state, 'buchi': automaton_state}, cost, done, new_info

    def simulate_step(self, state, buchi, action, is_eps=False):
        current_mdp_state, current_aut_state = self.mdp.get_state(), self.automaton.get_state()
        self.mdp.set_state(state)
        self.automaton.set_state(buchi)
        output = self.step(action, is_eps)
        self.mdp.set_state(current_mdp_state)
        self.automaton.set_state(current_aut_state)
        return output

    def find_min_accepting_cycles(self, start_state):
        visited = set()
        cycles = []
        # run a dfs
        def dfs(vertex, path):
            visited.add(vertex)
            #import pdb; pdb.set_trace()
            self.automaton.set_state(vertex)
            for edge in self.automaton.edges():
                #check if it's a valid edge
                neighbor = edge.child.id
                # if edge.child.accepting_vars.issubset(edge.parent.accepting_vars):
                #     edge.give_reward = False
                # else:
                #     edge.give_reward = True
                if neighbor == start_state:
                    path[vertex] = edge
                    # edge.give_reward = True  # always reward visiting the accept state
                    if path not in cycles:
                        # import pdb; pdb.set_trace()
                        print('Found cycle {}'.format(path))
                        cycles.append(deepcopy(path))
                else:
                    if neighbor in self.automaton.accepting_sets:
                        continue
                    if neighbor not in visited:
                        path[vertex] = edge
                        dfs(neighbor, deepcopy(path))
            visited.remove(vertex)
        dfs(start_state, {})
        return cycles
    
    def render(self, *args, **kw):
        return self.mdp.render(*args, **kw)
    
    def plot(self, *args, **kwargs):
        try:
            self.mdp.plot(*args, **kwargs)
        except:
            print('Cannot Print Policy')
    
    def test(self, *args, **kwargs):
        try:
            return self.mdp.test(*args, **kwargs)
        except:
            print('Cannot test Policy')
