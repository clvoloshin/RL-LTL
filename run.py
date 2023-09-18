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
from algs.ppo_continuous_2 import run_ppo_continuous_2, eval_agent
from algs.sac_learning import run_sac
import pickle as pkl
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
    save_dir = os.path.join(os.getcwd(), 'experiments', cfg['run_name'] + "_" + cfg["baseline"])
    results_dict = {}
    results_path = save_dir + '/results_dict.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if cfg["baseline"] == "all":
        baseline_types = ["ours", "pretrain_only", "cycler_only", "baseline"]
    else:
        baseline_types = [cfg["baseline"]]
    for bline in baseline_types:
        reward_sequence, eval_results = run_baseline(cfg, env, automaton, save_dir, bline)
        results_dict[bline + "_crewards"] = reward_sequence
        results_dict[bline + "_buchi"], results_dict[bline + "_mdp"], results_dict[bline + "_cr"] = eval_results[0], eval_results[1], eval_results[2]
        with open(results_path, 'wb') as f:
            pkl.dump(results_dict, f)
    print(cfg)

def run_baseline(cfg, env, automaton, save_dir, baseline_type, method="ppo"):
    if baseline_type == "ours":
        first_reward_type = 4
        second_reward_type = 2
        pretrain_trajs = cfg['ppo']['n_pretrain_traj']
        train_trajs = cfg['ppo']['n_traj']
    elif baseline_type == "pretrain_only":
        first_reward_type = 3
        second_reward_type = 1
        pretrain_trajs = cfg['ppo']['n_pretrain_traj']
        train_trajs = cfg['ppo']['n_traj']
    elif baseline_type == "cycler_only":
        first_reward_type = 2
        second_reward_type = 2
        pretrain_trajs = 0
        train_trajs = cfg['ppo']['n_pretrain_traj'] + cfg['ppo']['n_traj']
    elif baseline_type == "baseline":  # baseline method
        first_reward_type = 1
        second_reward_type = 1
        pretrain_trajs = 0
        train_trajs = cfg['ppo']['n_pretrain_traj'] + cfg['ppo']['n_traj']
    else:
        print("BASELINE TYPE NOT FOUND!")
        import pdb; pdb.set_trace()
    run_name = cfg['run_name'] + "_" + baseline_type + "_" + '_seed' + str(cfg['init_seed']) + '_lambda' + str(cfg['lambda']) + datetime.now().strftime("%m%d%y_%H%M%S")
    with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True), name=run_name) as run:
    # run_Q_STL(cfg, run, sim)
    # copt = ConstrainedOptimization(cfg, run, sim)
        total_crewards = []
        if method != 'ppo':
            import pdb; pdb.set_trace()
            sim = Simulator(env, automaton, cfg['lambda'], reward_type=first_reward_type)
            agent, total_crewards = run_Q_continuous(cfg, run, sim, visualize=cfg["visualize"], save_dir=save_dir)
        else:
        #run_sac(cfg, run, sim)
        #pretraining phase
            cfg['reward_type'] = first_reward_type
            sim = Simulator(env, automaton, cfg['lambda'], reward_type=first_reward_type)
            agent, pre_orig_crewards = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=True, visualize=cfg["visualize"],
                                                            save_dir=save_dir, save_model=True, n_traj=pretrain_trajs)
            total_crewards.extend(pre_orig_crewards)
            cfg['reward_type'] = second_reward_type
            if first_reward_type != second_reward_type:  # using our pretraining tactic, reset entropy.
                agent.reset_entropy()
            sim = Simulator(env, automaton, cfg['lambda'], reward_type=second_reward_type)
            agent, full_orig_crewards = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=True, visualize=cfg["visualize"],
                                                            save_dir=save_dir, save_model=True, agent=agent, n_traj=train_trajs)
            total_crewards.extend(full_orig_crewards)
        if baseline_type == "ours":
            traj_dir = save_dir + '/trajectories'
            if not os.path.exists(traj_dir):
                os.mkdir(traj_dir)
        else:
            traj_dir = None
        buchi_visits, mdp_reward, combined_rewards = eval_agent(cfg, run, sim, agent, save_dir=traj_dir)
        run.finish()
    return total_crewards, (buchi_visits, mdp_reward, combined_rewards)

def eval_policy(cfg, env, automaton, save_dir):
    sim = Simulator(env, automaton, cfg['lambda'], reward_type=1)
    

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