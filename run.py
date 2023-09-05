import hydra
from pathlib import Path
import wandb
import torch
import numpy as np
import os
from omegaconf import OmegaConf
from datetime import datetime
from envs.abstract_env import Simulator
from automaton import Automaton, AutomatonRunner
from algs.Q_value_iter_2 import run_value_iter
from algs.Q_continuous import run_Q_continuous
from algs.ppo_continuous_2 import run_ppo_continuous_2
from algs.sac_learning import run_sac
ROOT = Path(__file__).parent

@hydra.main(config_path=str(ROOT / "cfgs"))
def main(cfg):
    np.random.seed(cfg.init_seed)
    seeds = [np.random.randint(1e6) for _ in range(cfg.n_seeds)]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    env = hydra.utils.instantiate(cfg.env)
    automaton = AutomatonRunner(Automaton(**cfg['ltl']))
    # make logging dir for wandb to pull from, if necessary
    save_dir = os.path.join(os.getcwd(), 'experiments', cfg['logger']['dir_name'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    run_name = cfg['run_name'] + '_seed' + str(cfg['init_seed']) + '_lambda' + str(cfg['lambda']) + "_" + datetime.now().strftime("%m%d%y_%H%M%S")
    sim = Simulator(env, automaton, cfg['lambda'], reward_type=cfg['reward_type'])
    with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True), name=run_name) as run:
        # run_Q_STL(cfg, run, sim)
        # copt = ConstrainedOptimization(cfg, run, sim)
        if 'continuous' not in cfg['classes']:
            run_Q_continuous(cfg, run, sim, visualize=cfg["visualize"], save_dir=save_dir)
        else:
        #run_sac(cfg, run, sim)
            run_ppo_continuous_2(cfg, run, sim, to_hallucinate=True, visualize=cfg["visualize"], save_dir=save_dir)
        # run_value_iter(cfg, run, sim)
    print(cfg)

if __name__ == "__main__":
    main()

## G(F(g) & ~b & ~r & ~y)
#constrained_rew_fxn = {0: [env.automaton.edges(0, 1)[0], env.automaton.edges(0, 0)[0]], 1: [env.automaton.edges(1, 0)[0]]}

## G(F(y & X(F(r)))) & G~b
#constrained_rew_fxn = {0: [env.automaton.edges(0, 1)[0]], 1: [env.automaton.edges(1, 2)[0]], 2: [env.automaton.edges(2, 0)[0]]}
#import pdb; pdb.set_trace()
## F(G(y))
#constrained_rew_fxn = {1: [env.automaton.edges(1, 1)[0]]}

## F(r & XF(G(y)))
#constrained_rew_fxn = {2: [env.automaton.edges(2, 2)[0]]}  
#import pdb; pdb.set_trace()
## F(r & XF(g & XF(y))) & G~b
# constrained_rew_fxn = {2: [env.automaton.edges(2, 3)[0]], 3: [env.automaton.edges(3, 1)[0]], 1: [env.automaton.edges(1, 0)[0]], 0: [env.automaton.edges(0, 0)[0]]}  
# constrained_rew_fxn = {0: [env.automaton.edges(0, 0)[0]]}