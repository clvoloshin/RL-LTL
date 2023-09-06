from .base_envs.safety_gym import safety_gym
import gym

safety_gym_env = gym.make('Safexp-PointButton-v0')

# TODO: 
class SafetyGymWrapper:
    # the other functions are already implemented in the safety gym env
    # def label(self):
    # def reward(self):
    def __init__(self):
        self.original_env = gym.make('Safexp-PointButton-v0')
        self.original_env.task = 'goal' # reward is for reaching the goal
    
    def reset(self):
        return self.original_env.reset()

    def label(self, data):
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

    def render(self, states = [], save_dir=None, save_states=False):
        self.original_env.render(states, save_dir, save_states)

    def step(self, action):
        return self.original_env.step(action)


    
