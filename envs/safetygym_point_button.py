import safety_gym
import gym

#safety_gym_env = gym.make('Safexp-PointButton-v0')

# TODO: 
class SafetyGymWrapper:
    # the other functions are already implemented in the safety gym env
    # def label(self):
    # def reward(self):
    def __init__(self):
        self.original_env = gym.make('Safexp-PointButton1-v0')
        #self.original_env.task = 'goal' # reward is for reaching the goal
        self.observe_buttons = True # observe the button positions
        # self.constrain_button = True

        self.action_space = self.original_env.action_space
        self.observation_space = self.original_env.observation_space
        self.render_live = True
    
    def reset(self):
        self.close()
        state = self.original_env.reset()
        return self.state_wrapper(state), {}

    def close(self):
        self.original_env.close()
    
    def state_wrapper(self, state):
        return {
            'state': state,
            'data': self.original_env.data,
        }
    
    def get_info(self):
        return {}

    def label(self, state):
        data = state['data']
        # ! Pass the env.data to label instead of the state for safety gym envs
        # Why do we need signal here?
        signal, labels = {}, {}
        # reach the button
        for contact in data.contact[:data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.original_env.model.geom_id2name(g) for g in geom_ids])
            for idx in range(self.original_env.buttons_num):
                if any(n == f'button{idx}' for n in geom_names):
                    if any(n in self.original_env.robot.geom_names for n in geom_names):
                        labels.update({f'button_{idx}': 1})
        return labels, signal

    def render(self, states = [], save_dir=None, save_states=False):
        states = [s['state'] for s in states]
        self.original_env.render()

    def step(self, action):
        next_state, reward, done, info = self.original_env.step(action)
        return self.state_wrapper(next_state), reward, done, info
    
    def get_state(self):
        return self.state_wrapper(self.original_env.obs())

safety_gym_env = SafetyGymWrapper()
    
