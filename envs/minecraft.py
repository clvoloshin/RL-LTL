from .base_envs.slippery_grid import SlipperyGrid
from functools import partial
import numpy as np
from gym_minigrid.minigrid import Floor

# create a SlipperyGrid object
shape = [10, 10]
minecraft = SlipperyGrid(shape=shape, initial_state=[9, 2], slip_probability=0.)

# define the labellings
labels = np.empty([minecraft.shape[0], minecraft.shape[1]], dtype=object)
labels[0:10, 0:10] = 'safe'
labels[0:3, 5] = 'obstacle'
labels[2, 7:10] = 'obstacle'
labels[0][0] = labels[4][5] = labels[8][1] = labels[8][7] = 'grass'
labels[2][2] = labels[7][3] = labels[5][7] = labels[9][9] = 'wood'
labels[0][3] = labels[4][0] = labels[6][8] = labels[9][4] = 'iron'
# labels[6][1] = labels[6][5] = 
labels[4][9] = 'work_bench'
labels[2][4] = labels[9][0] = labels[7][7] = 'tool_shed'
labels[0][7] = 'gold'

for col in range(len(labels)):
    for row, label in enumerate(labels[col]):
        if label == 'safe': continue

        if label == 'obstacle':
            minecraft.grid.set(row, col, Floor('red'))
        # if label == 'grass':
        #     minecraft.grid.set(row, col, Floor('green'))
        # if label == 'wood':
        #     minecraft.grid.set(row, col, Floor('blue'))
        # if label == 'iron':
        #     minecraft.grid.set(row, col, Floor('purple'))
        if label == 'work_bench':
            minecraft.grid.set(row, col, Floor('blue'))
        # if label == 'tool_shed':
        #     minecraft.grid.set(row, col, Floor('blue'))
        if label == 'gold':
            minecraft.grid.set(row, col, Floor('yellow'))

# override the labels
minecraft.labels = labels

minecraft.cost = np.ones((shape[0], shape[1], 5))

def plot(self, dist=None, **kw):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import colors
    from scipy.stats.kde import gaussian_kde


    distinct_labels = np.unique(self.labels)
    labels_dic = {}
    label_indx = 0
    bounds = [-0.9]
    cmap = plt.get_cmap('gist_rainbow')
    for label in distinct_labels:
        labels_dic[label] = label_indx
        bounds.append(bounds[-1] + 1)
        label_indx += 1
    color_map = cmap(np.linspace(0, 1, len(distinct_labels)))
    cmap = colors.ListedColormap(color_map)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    labels_value = np.zeros([self.shape[0], self.shape[1]])
    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            labels_value[i][j] = labels_dic[self.state_label([i, j])]
    patches = [mpatches.Patch(color=color_map[i], label=list(distinct_labels)[i]) for i in
                range(len(distinct_labels))]
    plt.imshow(labels_value, interpolation='nearest', cmap=cmap, norm=norm)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # import pdb; pdb.set_trace()
    d0 = self.index_to_state(self.d0)

    # path_x, path_y = np.array(test_path).T
    # plt.scatter(path_y, path_x, c='lime', edgecolors='teal')
    plt.scatter(d0[1], d0[0], c='red', edgecolors='black')

    
    # Z, xedges, yedges = np.histogram2d(dist[:,:,0].reshape(-1), dist[:,:,1].reshape(-1))
    # plt.pcolormesh(xedges, yedges, Z.T, alpha=.1)

    if dist is not None:
        x = dist[:,:,0].reshape(-1)
        y = dist[:,:,1].reshape(-1)
        k = gaussian_kde(np.vstack([x, y]), bw_method=.05)
        yi, xi = np.mgrid[-.5:self.shape[0]-.5:x.size**0.5*1j,-.5:self.shape[1]-.5:y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        Z, xedges, yedges = np.histogram2d(x, y)

        plt.pcolormesh(yi, xi, zi.reshape(yi.shape), alpha=0.5)

    plt.annotate('s_0', (d0[1], d0[0]), fontsize=15, xytext=(20, 20), textcoords="offset points",
                    va="center", ha="left",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    
    plt.show()

    
def test(n_traj, sim, policy, J, c_min, map= lambda x: x):
    traj = []
    costs = []
    random_actions = []
    for n in range(n_traj):
        costs.append([])
        traj.append([sim.reset()])
        random_actions.append([])
        for t in range(100):
            try:
                sim.mdp.render()
            except:
                pass

            try:
                action = policy[map(traj[-1][-1])]
                random_act = 0
            except:
                # logger.warn('Policy sees new state: %d' % map(traj[-1][-1]))
                # import pdb; pdb.set_trace()
                random_act = 1
                action = np.random.choice(sim.mdp.action_space.n)

            next_state, cost, _, info = sim.step(action)

            traj[-1].append(next_state)
            costs[-1].append(1)
            random_actions[-1].append(random_act)
            # if info['goal']: break
    
    print([np.sum(cost) for cost in costs], np.mean([np.sum(cost) for cost in costs]), np.std([np.sum(cost) for cost in costs]))
    return np.array([[sim.mdp.index_to_state(sim.map[state_idx][0]) for state_idx in T] for T in traj])


minecraft.plot = partial(plot, minecraft)
minecraft.test = test
