from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add
import os
from copy import deepcopy
import PIL.Image as Image
from numpy import uint8

OBJECT_TO_IDX.update({'ghost': len(OBJECT_TO_IDX)})
OBJECT_TO_IDX.update({'coin': len(OBJECT_TO_IDX)})
SIZE = 50

class Ghost(WorldObj):
    def __init__(self):
        super().__init__('ghost', 'red')

    def can_overlap(self):
        return True

    def render(self, img):

        ghost = Image.open(os.path.join(os.getcwd(), 'envs/base_envs/layouts/pacman/ghost.png')).resize([SIZE, SIZE])
        im = Image.fromarray(img).convert('RGBA')
        im.paste(ghost, (50, 50), ghost)

        img[:] = np.array(im.convert('RGB'))

class Coin(WorldObj):
    def __init__(self):
        super().__init__('coin', 'yellow')

    def can_overlap(self):
        return True

    def render(self, img):

        coin = Image.open(os.path.join(os.getcwd(), 'envs/base_envs/layouts/pacman/coin.png')).resize([SIZE, SIZE])
        im = Image.fromarray(img).convert('RGBA')
        im.paste(coin, (50, 50), coin)

        img[:] = np.array(im.convert('RGB'))
    
    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, None)
        return True
        
class Pacman(MiniGridEnv):
    """
    Single-room square grid environment with moving obstacles
    """

    def __init__(
            self,
            layout='small1'
    ):

        file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layouts', layout) + '.py')
        self.layoutText = [line.strip() for line in file]
        file.close()
        self.width = len(self.layoutText[0])
        self.height = len(self.layoutText)
        self.labels = np.empty([self.width, self.height], dtype=object)
        self.labels[:] = 'safe'
        self.agent_state = []
        self.ghosts_state = []
        self.mdp_graph = {}
        self.agent_initial_state = []
        self.ghosts_initial_state = []
        self.current_state = []
        # directional actions
        self.action_map = [
            "right",
            "up",
            "left",
            "down",
            "stay",
        ]

        

        self.shortest_paths = {}

        size = self.width * self.height
        super().__init__(
            width = self.width,
            height=self.height,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

        
        self.beta = .6/20
        assert self.beta > 0

        # self.d0 = list(set([int(self.reset()) for x in range(10000)]))
        self.get_graph()
        self.reset()

        self.observation_space = gym.spaces.Discrete(( (self.width) * (self.height) )**2 )#* 2**len(self.food))
        self.action_space = gym.spaces.Discrete(len(self.action_map))
    
    def did_succeed(self, *args, **kw):
        return 0
    
    def reset(self):
        super().reset()

        self.agent_state = self.agent_initial_state
        self.ghosts_state = self.ghosts_initial_state.copy()
        self.current_state = [self.agent_state] + self.ghosts_state
        return self.get_state(), {}
    
    def render(self):
        super().render(highlight=False)


    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)

        # # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        self.tokens = []
        self.food = []
        self.ghosts = []

        food_counter = 0
        max_y = self.height - 1
        
        self.ghosts_initial_state = []
        self.ghosts_state = []
        for x in range(self.width):
            for y in range(self.height):
                try:
                    layout_char = self.layoutText[max_y - y][x]#[max_y - y][x]
                except:
                    import pdb; pdb.set_trace()
                if layout_char == '%':
                    self.labels[x][y] = 'obstacle'
                    self.grid.set(x, y, Wall())
                elif layout_char == '.':
                    pass
                    # self.labels[x][y] = 'token'
                    # self.tokens.append(Ball())
                    # self.grid.set(x, y, self.tokens[-1])
                elif layout_char == 'o':
                    self.labels[x][y] = 'food' + str(food_counter)
                    self.food.append(Coin())
                    food_counter += 1
                    self.food[-1].cell = self.location2cell([x, y])
                    self.food[-1].iseaten = False
                    self.grid.set(x, y, self.food[-1])
                elif layout_char == 'P':
                    self.agent_state = self.location2cell([x, y])
                    self.agent_initial_state = self.location2cell([x, y])
                elif layout_char in ['G']:
                    self.ghosts_state.append(self.location2cell([x, y]))
                    self.ghosts_initial_state.append(self.location2cell([x, y]))
                    self.ghosts.append(Ghost())
                    self.grid.set(x, y, self.ghosts[-1])
                    
                
        self.agent_pos = self.cell2location(self.agent_initial_state)
        self.agent_dir = 0
        self.mission = "Collect Boxes"

    def get_graph(self):
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                graph_vertex = self.location2cell([x, y])
                self.mdp_graph[graph_vertex] = []
                for a in range(len(self.action_map)):
                    if a == 0:
                        agent_next_location = list(map(add, self.cell2location(graph_vertex), [1, 0]))
                    elif a == 1:
                        agent_next_location = list(map(add, self.cell2location(graph_vertex), [0, -1]))
                    elif a == 2:
                        agent_next_location = list(map(add, self.cell2location(graph_vertex), [-1, 0]))
                    elif a == 3:
                        agent_next_location = list(map(add, self.cell2location(graph_vertex), [0, 1]))
                    elif a == 4:
                        agent_next_location = self.cell2location(graph_vertex)

                    if self.labels[agent_next_location[0]][agent_next_location[1]] is not None \
                            and 'obstacle' in self.labels[agent_next_location[0]][agent_next_location[1]]:
                        agent_next_location = self.cell2location(graph_vertex)

                    new_state = self.location2cell(agent_next_location)
                    if new_state not in self.mdp_graph[graph_vertex]:
                        self.mdp_graph[graph_vertex].append(new_state)
    
    def step(self, action_idx, ghosts_moving=1):

        self.step_count = 0
        action = self.action_map[action_idx]
        
        self.agent_state = self.current_state[0]
        self.ghosts_state = self.current_state[1:]

        # if all([x.iseaten for x in self.food]):
        #     next_state = self.get_state()
        #     cost = 1
        #     done = True
        #     info = {'state': self.current_state}
        #     return next_state, cost, done, info

        prev_ghost_pos = deepcopy(self.ghosts_state)
        # agent movement dynamics:
        if action == 'right':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [1, 0]))
            self.agent_dir = 0
        elif action == 'up':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [0, -1]))
            self.agent_dir = 3
        elif action == 'left':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [-1, 0]))
            self.agent_dir = 2
        elif action == 'down':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [0, 1]))
            self.agent_dir = 1
        elif action == 'stay':
            agent_next_location = self.cell2location(self.agent_state)

        # check for obstacles
        if self.labels[agent_next_location[0]][agent_next_location[1]] is not None \
                and 'obstacle' in self.labels[agent_next_location[0]][agent_next_location[1]]:
            agent_next_location = self.cell2location(self.agent_state)

        # update agent state
        # obs, reward, done, info = MiniGridEnv.step(self, action_idx)
        self.agent_state = self.location2cell(agent_next_location)
        self.agent_pos = agent_next_location

        # how ghosts chase the agent
        
        if ghosts_moving:
            for i in range(len(self.ghosts_state)):
                # with probability of 40% the ghosts chase the pacman
                if np.random.uniform() < .4:
                    
                    start, finish = self.ghosts_state[i], self.agent_state
                    if (start, finish) in self.shortest_paths: 
                        tmp = self.shortest_paths[(start, finish)] 
                    else: 
                        tmp = self.find_shortest_path(self.mdp_graph,
                                                    self.ghosts_state[i],
                                                    self.agent_state)
                        self.shortest_paths[(start, finish)] = tmp
                        # print(len(self.shortest_paths))
                    
                    if tmp is not None:
                        try:
                            self.ghosts_state[i] = tmp[1]
                        except:
                            self.ghosts_state[i] = tmp[0]
                    else:
                        import pdb; pdb.set_trace()

                # with probability of 60% the ghosts take a random action
                else:
                    idx = np.random.randint(4)
                    acts = [[1, 0],
                    [-1, 0],
                    [0, -1],
                    [1, 0]]
                    random_act = acts[idx]
                    ghost_i_next_location = list(map(add, self.cell2location(self.ghosts_state[i]), random_act))
                    # check for obstacles
                    if self.labels[ghost_i_next_location[0]][ghost_i_next_location[1]] is not None \
                            and 'obstacle' in self.labels[ghost_i_next_location[0]][ghost_i_next_location[1]]:
                        ghost_i_next_location = self.cell2location(self.ghosts_state[i])

                    # update ghost state
                    self.ghosts_state[i] = self.location2cell(ghost_i_next_location)

            # Update obstacle positions
            # update for rendering
            for new_ghost_state, prev_ghost_state, ghost in zip(self.ghosts_state, prev_ghost_pos, self.ghosts):
                
                try:
                    self.grid.set(*self.cell2location(prev_ghost_state), None)
                    self.grid.set(*self.cell2location(new_ghost_state), ghost)
                except:
                    import pdb; pdb.set_trace()

        # return the MDP state
        mdp_state = [self.agent_state] + self.ghosts_state
        self.current_state = mdp_state

        for i, food in enumerate(self.food):
            
            if food.iseaten:
                pass
            if self.agent_state == food.cell:
                food.iseaten = True
                food.toggle(self, self.cell2location(food.cell))
            elif (not food.iseaten) and (food.cell not in self.ghosts_state):
                self.grid.set(*self.cell2location(food.cell), food)
            
        next_state = self.get_state()

        # prev_state = self.state_to_index(prev_state)
        try:
            cost = 1
        except:
            import pdb; pdb.set_trace()
        done = 'ghost' in self.label(next_state)[0] #all([x.iseaten for x in self.food])
        
        info = {'state': self.current_state}

        return next_state, cost, done, info

    # def step(self, action):
    #     # Invalid action
    #     if action >= self.action_space.n:
    #         action = 0

    #     # Check if there is an obstacle in front of the agent
    #     front_cell = self.grid.get(*self.front_pos)
    #     not_clear = front_cell.type == 'wall'

    #     # Update obstacle positions
    #     for i_obst in range(len(self.obstacles)):
    #         old_pos = self.obstacles[i_obst].cur_pos
    #         top = tuple(map(add, old_pos, (-1, -1)))

    #         try:
    #             self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
    #             self.grid.set(*old_pos, None)
    #         except:
    #             pass

    #     # Update the agent's position/direction
    #     obs, reward, done, info = MiniGridEnv.step(self, action)

    #     # If the agent tried to walk over an obstacle or wall
    #     if action == self.actions.forward and not_clear:
    #         reward = -1
    #         done = True
    #         return obs, reward, done, info

    #     return obs, reward, done, info

    def state_label(self, state):
        location = self.cell2location(state[0])
        if state[0] in state[1:]:
            return 'ghost'
        else:
            return self.labels[location[0]][location[1]]
    
    def state_to_index(self, state):
        base_state = state[:-len(self.food)]
        food_state = state[len(base_state):]
        idx0 = np.sum(np.array([x for x in base_state]) * (self.width*self.height)**np.arange(len(base_state))) 
        idx1 = np.sum(np.array([x for x in food_state]) * (2)**np.arange(len(food_state))[::-1])
        idx = idx0 #* (2**len(food_state)) + idx1
        return idx
    
    def index_to_state(self, index):
        
        foods = [int(x) == 1 for x in format(index % 2**len(self.food), '0%db' % len(self.food))]
                
        index = index #// 2**len(self.food)


        state = np.zeros(2)
        state[0] = int(index % (self.width*self.height))
        state[1] = int((index % (self.width*self.height)**2)  // (self.width*self.height))

        return [int(x) for x in state] #+ foods
    
    def cost_shaping(self, prev_index, cur_index, action, automaton_movement, accepting_state_reached, rejecting_state_reached):
        
        cost = 1

        if (automaton_movement and not rejecting_state_reached):
            cost = 0
        
        if accepting_state_reached:
            cost = 0
        
        if rejecting_state_reached:
            cost = 10000

        return cost, False

    def location2cell(self, state):
        return state[0] + state[1] * self.width

    def cell2location(self, cell_num):
        return [cell_num % self.width, int(cell_num / self.width)]

    def find_shortest_path(self, graph, start, end, path=None):
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return path
        if start not in graph.keys():
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = self.find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
    
    def label(self, state_idx):
        # labels = self.labels.get(self.state, None)
        state = self.index_to_state(state_idx)
        labels = [self.state_label(state)]
        # if 'ghost' in labels: 
        #     import pdb; pdb.set_trace()
        # if 'food0' in labels:
        #     import pdb; pdb.set_trace()
        if labels is None:
            return {}, {}
        else:
            return {label:1 for label in labels}, {}
    
    def get_cost(self, state, action):
        return 1
        
    def set_state(self, state):
        assert (state >= 0) and (state <= self.observation_space.n * (2**len(self.food))), 'Setting Environment to invalid state'
        
        new_state = self.index_to_state(state)
        self.current_state = new_state[:-len(self.food)]
        for food, val in zip(self.food, new_state[len(self.current_state):]):
            food.iseaten = val
    
    def get_state(self):
        idx = self.state_to_index(self.current_state + [x.iseaten for x in self.food])
        return idx 