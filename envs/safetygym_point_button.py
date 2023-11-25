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
        return {}

    def compute_rho(self):
        # compute robustness
        # get distances to buttons
        pass

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
        current_pos = self.original_env.task.agent.pos
        #for

    def render(self, states = [], save_dir=None, save_states=False):
        states = [s['state'] for s in states]
        self.original_env.render()

    def step(self, action):
        next_state, reward, cost, terminated, truncated, info = self.original_env.step(action)
        self.state = next_state
        self.current_cost = info
        if "cost_gremlins" not in self.current_cost:
            self.current_cost["cost_gremlins"] = 0
        self.info = info
        # import pdb; pdb.set_trace()
        return self.state_wrapper(next_state), 0, terminated, info
    
    def get_state(self):
        return self.state_wrapper(self.state)

safety_gym_env = SafetyGymWrapper()
    
