from .base_envs.pacman import Pacman
from functools import partial
import numpy as np
from PIL import Image
import os
import imageio


pacman = Pacman()
pacman.cost = np.array(1)

def plot(self, ims=None, dir=None, **kw):
    for i, traj in enumerate(ims[0][::10]):
        imageio.mimsave(os.path.join(dir, 'traj_%d.gif' % i), traj, fps=5)
        # traj[0].save(, save_all=True, append_images=traj[1:], loop=0)

def test(n_traj, sim, policy, J, c_min, map= lambda x: x, collect_ims=False):
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


pacman.plot = partial(plot, pacman)
pacman.test = test