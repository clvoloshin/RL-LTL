import argparse
import os
import logger
import yaml
import torch
import numpy as np
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from experiment_tools.factory import setup_params

# from algs.ppo_ltl import run_ppo_ltl
from algs.min_ppo import run_PPO
from envs.abstract_env import SimulatorMDP

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor")

def main(seed, param, to_redo):
    logger.info('*'*20)

    dir = os.path.join(param['logger']['dir_name'], 'ours_ppo', 'experiment_%05.f' % (seed) )
    torch.manual_seed(seed)
    np.random.seed(seed)
    param['env']['file'] = param['classes']['continuous']
    env = setup_params(param)
    sim = SimulatorMDP(env)
    logger.configure(name=dir)
    run_PPO(param, sim)
    

if __name__ == '__main__':
    # Local:
    # python run.py chain.yaml

    parser = argparse.ArgumentParser(description='Run Experiment')

    parser.add_argument('cfg', help='config file', type=str)
    parser.add_argument('-r', '--restart', action='store_true')
    args = parser.parse_args()

    assert args.cfg.endswith('.yaml'), 'Must be yaml file'
    with open(os.getcwd() + '/cfgs/{0}'.format(args.cfg), 'r') as f:
        param = yaml.load(f, Loader=Loader)

    np.random.seed(param['init_seed'])
    seeds = [np.random.randint(1e6) for _ in range(param['n_seeds'])]

    for seed in seeds:
        print('*' * 20)
        logger.Logger.set_level(logger,logger.DEBUG)
        logger.info("Seed = {}".format(float(seed)))
        main(seed, param, args.restart)