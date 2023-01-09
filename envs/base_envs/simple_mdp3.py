from ..abstract_env import AbstractEnv
import gym
import numpy as np
from PIL import Image
import imageio
import os

class SimpleMDP(AbstractEnv, gym.Env):
    def __init__(self, **kw):
        
        self.observation_space = gym.spaces.Discrete(kw['observation_space'])
        self.action_space = gym.spaces.Discrete(kw['action_space'])
        self.d0 = kw['d0']
        self.T = kw['T']
        self.cost = np.array(kw['cost'])
        self.labels = kw['labels']

        self.current_state = self.d0
        self.beta = np.min([[min(Ts_.values()) for a, Ts_ in Tas_.items()] for s, Tas_ in self.T.items()])
    
    def label(self, state):
        labels = self.labels.get(state, None)
        if labels is None:
            return {}
        else:
            return {label:1 for label in labels}
        
    def set_state(self, state):
        assert (state >= 0) and (state <= self.observation_space.n), 'Setting Environment to invalid state'
        self.current_state = state
    
    def get_state(self):
        return self.current_state
    
    def get_cost(self, state, action):
        return self.cost[state, action]
    
    def reset(self):
        self.current_state = self.d0
        return self.current_state
    
    def step(self, action):
        
        probs = self.T.get(self.current_state, 0).get(action, 0)
        try:
            next_state = np.random.choice(list(probs.keys()), p=list(probs.values()))
        except:
            import pdb; pdb.set_trace()
        cost = self.cost[self.current_state, action]
        done = False
        info = {'state': next_state}
        self.current_state = next_state

        return next_state, cost, done, info
    
    def plot(self, ims=None, dir=None, **kw):
        for i, traj in enumerate(ims[0][::10]):
            imageio.mimsave(os.path.join(dir, 'traj_%d.gif' % i), traj, fps=5)
            # traj[0].save(, save_all=True, append_images=traj[1:], loop=0)
    
    def cost_shaping(self, prev_index, cur_index, action, automaton_movement, accepting_state_reached, rejecting_state_reached):
        
        cost = 1

        # if (automaton_movement and not rejecting_state_reached):
        #     cost = .5
        
        if accepting_state_reached:
            cost = 0
        
        if rejecting_state_reached:
            cost = 100

        return cost, False
    
    def P_dense(self):
        P = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n))
        for s in range(self.observation_space.n):
            for a in range(self.action_space.n):
                Ts = self.T.get(s, {}).get(a, {})
                if len(Ts):
                    P[s,a,list(Ts.keys())] = list(Ts.values())
        return P

    def test(self, n_traj, sim, policy, J, c_min, map= lambda x: x, collect_ims=False, **kw):
        traj = []
        costs = []
        random_actions = []
        failure = 0
        success = 0
        ims = []
        for n in range(n_traj):
            ims.append([])
            costs.append([])
            traj.append([sim.reset()])
            random_actions.append([])
            if collect_ims:
                try:
                    ims[-1].append(Image.fromarray(sim.mdp.render(mode='rgb_array', highlight=False)))
                except:
                    pass
            for t in range(100):
                # try:
                #     action = policy[map(traj[-1][-1])]
                #     random_act = 0
                # except:
                #     # logger.warn('Policy sees new state: %d' % map(traj[-1][-1]))
                #     # import pdb; pdb.set_trace()
                #     random_act = 1
                #     action = np.random.choice(sim.mdp.action_space.n)
                action, random_act = policy.sample(map(traj[-1][-1]))

                next_state, cost, done, info = sim.step(action)

                traj[-1].append(next_state)
                costs[-1].append(1)
                random_actions[-1].append(random_act)
                if collect_ims:
                    try:
                        ims[-1].append(Image.fromarray(sim.mdp.render(mode='rgb_array', highlight=False)))
                    except:
                        pass

                if info['fail']: 
                    failure += 1
                    break
                # if done: 
                #     success += 1 
                #     break
                
        
        return ims, n_traj-failure, failure, [np.sum(cost) for cost in costs], np.mean([np.sum(cost) for cost in costs]), np.std([np.sum(cost) for cost in costs])



