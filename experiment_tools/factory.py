from logger import warn
import inspect
import gymnasium as gym
import importlib


def env_factory(param):
    """
        Load Environment
    """
    if "class" in param['env']:
        path = param['env']['class'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        env_class = getattr(importlib.import_module(module_name), class_name)
        env = env_class(**param['env'])
    elif "file" in param['env']:
        path = param['env']['file'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        env = getattr(importlib.import_module(module_name), class_name)
    else:
        raise ValueError("The configuration should specify the env __class__")

    return env
    
def setup_params(param):
    
    env = env_factory(param)

    return env