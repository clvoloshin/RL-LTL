import argparse
import os
import logger
import yaml
import torch
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from experiment_tools.factory import setup_params

# from algs.ppo_ltl import run_ppo_ltl
from algs.Q_learning import run_Q_learning
from algs.ppo import run_PPO
import numpy as np
from envs.abstract_env import Simulator
from automaton import Automaton, AutomatonRunner

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor")

def main(seed, param, to_redo):
    automaton = AutomatonRunner(Automaton(**param['ltl']))
    logger.info('*'*20 + '\tLTL: %s' % automaton.automaton.formula)

    dir = os.path.join(param['logger']['dir_name'], 'ours_q', 'experiment_%05.f' % (seed) )
    if ('discrete' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
        torch.manual_seed(seed)
        np.random.seed(seed)
        param['env']['file'] = param['classes']['discrete']
        env = setup_params(param)
        sim = Simulator(env, automaton)

        # Simple check to see if observation_space space is discrete
        is_discrete_obs_space = False
        try:
            sim.observation_space['mdp'].n
            is_discrete_obs_space = True
        except:
            pass
        
        logger.configure(name=dir)
        run_Q_learning(param, sim, False, not is_discrete_obs_space, to_hallucinate=True)
    
    dir = os.path.join(param['logger']['dir_name'], 'baseline_q', 'experiment_%05.f' % (seed) )
    if ('discrete' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
        torch.manual_seed(seed)
        np.random.seed(seed)
        param['env']['file'] = param['classes']['discrete']
        env = setup_params(param)
        sim = Simulator(env, automaton)

        # Simple check to see if observation_space space is discrete
        is_discrete_obs_space = False
        try:
            sim.observation_space['mdp'].n
            is_discrete_obs_space = True
        except:
            pass
        
        logger.configure(name=dir)
        run_Q_learning(param, sim, False, not is_discrete_obs_space, to_hallucinate=False)

    dir = os.path.join(param['logger']['dir_name'], 'ours_ppo', 'experiment_%05.f' % (seed) )
    if ('continuous' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
        torch.manual_seed(seed)
        np.random.seed(seed)
        param['env']['file'] = param['classes']['continuous']
        env = setup_params(param)
        sim = Simulator(env, automaton)

        # Simple check to see if observation_space space is discrete
        is_discrete_obs_space = False
        try:
            sim.observation_space['mdp'].n
            is_discrete_obs_space = True
            # PPO for discrete state space: NOT IMPLEMENTED
            return
        except:
            pass
        
        logger.configure(name=dir)
        run_PPO(param, sim, False, not is_discrete_obs_space, to_hallucinate=True)
    
    dir = os.path.join(param['logger']['dir_name'], 'baseline_ppo', 'experiment_%05.f' % (seed) )
    if ('continuous' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
        torch.manual_seed(seed)
        np.random.seed(seed)
        param['env']['file'] = param['classes']['continuous']
        env = setup_params(param)
        sim = Simulator(env, automaton)

        # Simple check to see if observation_space space is discrete
        is_discrete_obs_space = False
        try:
            sim.observation_space['mdp'].n
            is_discrete_obs_space = True
            # PPO for discrete state space: NOT IMPLEMENTED
            return
        except:
            pass
        
        logger.configure(name=dir)
        run_PPO(param, sim, False, not is_discrete_obs_space, to_hallucinate=False)
    
    # dir = os.path.join(param['logger']['dir_name'], 'baseline_only_init', 'experiment_%05.f' % (seed) )
    # if to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir)):
    #     logger.configure(name=dir)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     param['env']['file'] = param['classes']['continuous']
    #     env = setup_params(param)
    #     sim = Simulator(env, automaton)

    #     # Simple check to see if observation_space space is discrete
    #     is_discrete_obs_space = False
    #     try:
    #         sim.observation_space['mdp'].n
    #         is_discrete_obs_space = True
    #     except:
    #         pass
        
    #     run_ppo_continuous(param, sim, False, to_hallucinate=False)

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
        # param['logger']['name'] = #'experts_and_'+param['MCTS']['bandit_strategy'] if param['experiment']['experts'] else 'policy_and_'+param['MCTS']['bandit_strategy']
        # logger.configure(name=os.path.join(param['logger']['dir_name'], 'experiment_%05.f' % (seed) ))
        logger.Logger.set_level(logger,logger.DEBUG)
        logger.info("Seed = {}".format(float(seed)))
        main(seed, param, args.restart)