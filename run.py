import hydra
from pathlib import Path
import wandb
import os
import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
from experiment_tools.factory import setup_params

# from algs.ppo_ltl import run_ppo_ltl
from algs.Q_learning import run_Q_learning
import numpy as np
from envs.abstract_env import Simulator
from automaton import Automaton, AutomatonRunner

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor")
ROOT = Path(__file__).parent

@hydra.main(config_path=str(ROOT / "cfgs"))
def main(cfg):
    np.random.seed(cfg.init_seed)
    seeds = [np.random.randint(1e6) for _ in range(cfg.n_seeds)]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    env = hydra.utils.instantiate(cfg.env)
    automaton = AutomatonRunner(Automaton(**cfg['ltl']))

    with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True)) as run:
        sim = Simulator(env, automaton)

        # Simple check to see if observation_space space is discrete
        is_discrete_obs_space = False
        # try:
        #     sim.observation_space['mdp'].n
        #     is_discrete_obs_space = True
        # except:
        #     pass
        
        run_Q_learning(cfg, run, sim, False, not is_discrete_obs_space, to_hallucinate=True)


if __name__ == '__main__':
    # Local:
    # python run.py chain.yaml

    main()