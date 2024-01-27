import numpy as np
import safety_gymnasium
from gymnasium.spaces import Box
#safety_gym_env = gym.make('Safexp-PointButton-v0')


class SafetyGymWrapper:
    # the other functions are already implemented in the safety gym env
    # def label(self):
    # def reward(self):
    def __init__(self, render_mode=None):
        if render_mode == "None":
            render_mode = None
        self.original_env = safety_gymnasium.make('SafetyPointButton1-v0', render_mode=render_mode)
        #self.original_env = ButtonEnvWrapper(config={"agent_name": 'Point'})
        #self.original_env.task = 'goal' # reward is for reaching the goal
        self.observe_buttons = True # observe the button positions
        # self.constrain_button = True

        self.action_space = self.original_env.action_space
        # import pdb; pdb.set_trace()
        self.observation_space = self.construct_obs_space()
        self.render_live = True
        self.current_cost = None
        self.reset()
        # import pdb; pdb.set_trace()
        self.rho_alphabet = ['button0', 'button1', 'button2', 'button3', 'gremlin']
        self.rho_min = -5.66 # hardcoded, but the max distance to anything in the boxed-in env
        self.rho_max = 0 # the closest you can get to a region is 0

    def construct_obs_space(self):
        new_low = np.append(self.original_env.observation_space.low, np.array([-float('inf'), -float('inf'), -float('inf')]))
        new_high = np.append(self.original_env.observation_space.low, np.array([float('inf'), float('inf'), float('inf')]))
        return Box(low=new_low, high=new_high, dtype=np.float32)

    def reset(self, options=None):
        state, info = self.original_env.reset()
        # import pdb; pdb.set_trace()
        self.state = state
        self.info = info
        return self.state_wrapper(state), info
    
    def state_wrapper(self, state):
        state = np.append(state, self.original_env.task.agent.pos)
        return {
            'state': state,
            'data': self.get_current_labels()
        }
    
    def get_info(self):
        return self.info
        
    def get_current_labels(self):
        labels = {}
        if self.current_cost is None:
            return labels
        if self.current_cost["cost_gremlins"] > 0:
            labels.update({"gremlin": 1})  
        labels.update( self.buttons_contacted())
        return labels

    def buttons_contacted(self):
        """Checks which button was just contacted."""
        button_labels = {}
        task = self.original_env.task
        for contact in task.data.contact[: task.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([task.model.geom(g).name for g in geom_ids])
            for button_num in range(task.buttons.num):
                # pylint: disable-next=no-member
                if any(n == f'button{button_num}' for n in geom_names) and any(
                    n in task.agent.body_info.geom_names for n in geom_names
                ):
                    button_labels[f'button{button_num}'] = 1
        return button_labels

    def label(self, state):
        labels = state['data']
        # ! Pass the env.data to label instead of the state for safety gym envs
        # Why do we need signal here?
        # reach the button
        return labels, {}
    
    def compute_rho(self):
        all_robustness_vals = []
        current_pos = self.original_env.task.agent.pos
        button_positions = self.original_env.task.buttons.pos
        #for each button, get the distance to the button
        for button_pos in button_positions:
            dist = -1 * np.linalg.norm(current_pos - button_pos) # we want to minimize the distance, so give it as a negative reward
            all_robustness_vals.append(dist)
        try:
            gremlin_positions = self.original_env.task.gremlins.pos
        except:
            gremlin_positions = []
        gremlin_dists = [np.linalg.norm(current_pos - gpos) for gpos in gremlin_positions]
        if len(gremlin_dists) == 0:
            all_robustness_vals.append(0) # give a positive value, i.e. we're satisfying this part of the spec
        all_robustness_vals.append(-1 * min(gremlin_dists)) # want the closest gremlin distance here
        return (np.array(all_robustness_vals)) #** 2

    def render(self, states = [], save_dir=None, save_states=False):
        states = [s['state'] for s in states]
        self.original_env.render()

    def step(self, action):
        next_state, reward, cost, terminated, truncated, info = self.original_env.step(action)
        self.state = next_state
        self.current_cost = info
        if "cost_gremlins" not in self.current_cost:
            self.current_cost["cost_gremlins"] = 0
        if info["cost_hazards"] > 0:
           new_reward = 0.5
        else:
            new_reward = 0
        self.info = info
        self.info["rhos"] = self.compute_rho()
        # if abs(reward) > 0.1:
        #     import pdb; pdb.set_trace()
        # can set reward to reward * 100 to debug
        # import pdb; pdb.set_trace()
        return self.state_wrapper(next_state), new_reward, terminated, self.info
    
    def get_state(self):
        return self.state_wrapper(self.state)

safety_gym_env = SafetyGymWrapper()
    
