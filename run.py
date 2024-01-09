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
from algs.Q_continuous import run_Q_continuous, eval_q_agent
from algs.Q_discrete import run_Q_discrete, eval_q_agent
from algs.ppo_continuous_2 import run_ppo_continuous_2, eval_agent, PPO
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
        baseline_types = ["ours", "baseline", "pretrain_only", "cycler_only"]
    else:
        baseline_types = [cfg["baseline"]]
    if 'continuous' not in cfg['classes']:
        method = "q_learning"
    else:
        method = "ppo"
    for bline in baseline_types:
        reward_sequence, buchi_traj_sequence, mdp_traj_sequence, eval_results = run_baseline(cfg, env, automaton, save_dir, bline, method=method)
        results_dict[bline + "_crewards"] = reward_sequence
        results_dict[bline + "_btrajs"] = buchi_traj_sequence
        results_dict[bline + "_mdptrajs"] = mdp_traj_sequence
        results_dict[bline + "_buchi"], results_dict[bline + "_mdp"], results_dict[bline + "_cr"] = eval_results[0], eval_results[1], eval_results[2]
        with open(results_path, 'wb') as f:
            pkl.dump(results_dict, f)
    print(cfg)

def run_baseline(cfg, env, automaton, save_dir, baseline_type, method="ppo"):
    if baseline_type == "ours":
        first_reward_type = 4
        second_reward_type = 2
        pretrain_trajs = cfg[method]['n_pretrain_traj']
        train_trajs = cfg[method]['n_traj']
        to_hallucinate = True
    elif baseline_type == "pretrain_only":
        first_reward_type = 3
        second_reward_type = 1
        pretrain_trajs = cfg[method]['n_pretrain_traj']
        train_trajs = cfg[method]['n_traj']
        to_hallucinate = True
    elif baseline_type == "cycler_only":
        first_reward_type = 2
        second_reward_type = 2
        pretrain_trajs = 0
        train_trajs = cfg[method]['n_pretrain_traj'] + cfg[method]['n_traj']
        to_hallucinate = True
    elif baseline_type == "baseline":  # baseline method
        first_reward_type = 1
        second_reward_type = 1
        pretrain_trajs = 0
        train_trajs = cfg[method]['n_pretrain_traj'] + cfg[method]['n_traj']
        to_hallucinate = True
    elif baseline_type == "ppo_only":  # baseline method
        first_reward_type = 1
        second_reward_type = 1
        pretrain_trajs = 0
        train_trajs = cfg[method]['n_pretrain_traj'] + cfg[method]['n_traj']
        to_hallucinate = False
    elif baseline_type == "quant":  # baseline method
        first_reward_type = 0
        second_reward_type = 0
        pretrain_trajs = 0
        train_trajs = cfg[method]['n_pretrain_traj'] + cfg[method]['n_traj']
        to_hallucinate = False
    elif baseline_type == "eval": # evaluate an existing model
        assert cfg["load_path"] is not None
        assert cfg["load_path"] != ""
    else:
        print("BASELINE TYPE NOT FOUND!")
        import pdb; pdb.set_trace()
    run_name = cfg['run_name'] + "_" + baseline_type + "_" + '_seed' + str(cfg['init_seed']) + '_lambda' + str(cfg['lambda']) + "_" + datetime.now().strftime("%m%d%y_%H%M%S")
    # run_Q_STL(cfg, run, sim)
    # copt = ConstrainedOptimization(cfg, run, sim)
    total_crewards = []
    total_buchis = []
    total_mdps = []
    if baseline_type != "eval":
        with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True), name=run_name) as run:
            if method != 'ppo':
                # import pdb; pdb.set_trace()
                if pretrain_trajs != 0:
                    sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=first_reward_type)
                    agent, pre_orig_crewards, buchi_trajs, mdp_trajs = run_Q_continuous(cfg, run, sim, visualize=cfg["visualize"], save_dir=save_dir, n_traj=pretrain_trajs)
                    total_crewards.extend(pre_orig_crewards)
                    total_buchis.extend(buchi_trajs)
                    total_mdps.extend(mdp_trajs)
                    if first_reward_type != second_reward_type:  # using our pretraining tactic, reset entropy.
                        agent.reset_entropy()
                    if baseline_type == "ours":
                        agent.reset_entropy()
                if train_trajs != 0:
                    sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=second_reward_type)
                    agent, full_orig_crewards, buchi_trajs, mdp_trajs = run_Q_continuous(cfg, run, sim, visualize=cfg["visualize"], save_dir=save_dir, agent=agent, n_traj=train_trajs)
                    total_crewards.extend(full_orig_crewards)
                    total_buchis.extend(buchi_trajs)
                    total_mdps.extend(mdp_trajs)
            else:
                #pretraining phase
                # TODO: Get this to run modularly like the PPO block of code.
                sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=first_reward_type)
                agent, pre_orig_crewards, buchi_trajs, mdp_trajs = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=to_hallucinate, visualize=cfg["visualize"],
                                                                save_dir=save_dir, save_model=True, n_traj=pretrain_trajs)
                total_crewards.extend(pre_orig_crewards)
                total_buchis.extend(buchi_trajs)
                total_mdps.extend(mdp_trajs)
                if first_reward_type != second_reward_type:  # using our pretraining tactic, reset entropy.
                    agent.reset_entropy()
                if baseline_type == "ours":
                    agent.reset_entropy()
                sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=second_reward_type)
                agent, full_orig_crewards, buchi_trajs, mdp_trajs = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=to_hallucinate, visualize=cfg["visualize"],
                                                                save_dir=save_dir, save_model=True, agent=agent, n_traj=train_trajs)
                total_crewards.extend(full_orig_crewards)
                total_buchis.extend(buchi_trajs)
                total_mdps.extend(mdp_trajs)
            if baseline_type == "ours":
                traj_dir = save_dir + '/trajectories'
                if not os.path.exists(traj_dir):
                    os.mkdir(traj_dir)
            else:
                traj_dir = None
            run.finish()
    else:
        # in evaluation mode
        sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=0)
        traj_dir = None
        agent = PPO(sim.observation_space, 
        sim.action_space, 
        cfg['gamma'], 
        cfg, 
        False,
        model_path=cfg['load_path'])
        # define agent here and load the existing model path (need to import from policy files)
    if method != 'ppo':
        buchi_visits, mdp_reward, combined_rewards = eval_q_agent(cfg, sim, agent, save_dir=traj_dir)
    else:
        buchi_visits, mdp_reward, combined_rewards = eval_agent(cfg, sim, agent, save_dir=traj_dir)
    return total_crewards, total_buchis, total_mdps, (buchi_visits, mdp_reward, combined_rewards)
    

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