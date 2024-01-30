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
from algs.ppo_continuous_2 import run_ppo_continuous_2, eval_agent, PPO
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
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    baseline = cfg["baseline"]
    if 'continuous' not in cfg['classes']:
        method = "q_learning"
    else:
        method = "ppo"
    for seed in seeds:
        results_dict = {}
        results_path = save_dir + '/results_dict_{}.pkl'.format(seed)
        np.random.seed(seed)
        reward_sequence, buchi_traj_sequence, mdp_traj_sequence, test_reward_sequence, test_buchi_sequence, test_mdp_sequence, eval_results = run_baseline(cfg, env, automaton, save_dir, baseline, seed, method=method)
        results_dict["crewards"] = reward_sequence
        results_dict["btrajs"] = buchi_traj_sequence
        results_dict["mdptrajs"] = mdp_traj_sequence
        results_dict["test_creward_values"] = test_reward_sequence
        results_dict["test_b_visits"] = test_buchi_sequence
        results_dict["test_mdp_rewards"] = test_mdp_sequence
        results_dict["buchi_eval"], results_dict["mdp_eval"], results_dict["cr_eval"] = eval_results[2], eval_results[3], eval_results[4]
        results_dict["evaltime_test_buchi_visits"], results_dict["evaltime_test_mdp_reward"] = eval_results[0], eval_results[1]
        with open(results_path, 'wb') as f:
            pkl.dump(results_dict, f)
    print(cfg)

def run_baseline(cfg, env, automaton, save_dir, baseline_type, seed, method="ppo"):
    if baseline_type == "ours":
        reward_type = 2
        to_hallucinate = True
    elif baseline_type == "no_mdp":
        reward_type = 3
        to_hallucinate = True
    elif baseline_type == "baseline":  # baseline method
        reward_type = 1
        pretrain_trajs = 0
        to_hallucinate = True
    elif baseline_type == "ppo_only":  # baseline method
        reward_type = 1
        pretrain_trajs = 0
        to_hallucinate = False
    elif baseline_type == "quant":  # baseline method
        reward_type = 0
        pretrain_trajs = 0
        to_hallucinate = True
    elif baseline_type == "eval": # evaluate an existing model
        assert cfg["load_path"] is not None
        assert cfg["load_path"] != ""
    else:
        print("BASELINE TYPE NOT FOUND!")
        import pdb; pdb.set_trace()
    train_trajs = cfg[method]['n_traj']
    run_name = cfg['run_name'] + "_" + baseline_type + "_" + '_seed' + str(seed) + '_lambda' + str(cfg['lambda']) + "_" + datetime.now().strftime("%m%d%y_%H%M%S")
    # run_Q_STL(cfg, run, sim)
    # copt = ConstrainedOptimization(cfg, run, sim)
    total_crewards = []
    total_buchis = []
    total_mdps = []
    total_test_crewards = []
    total_test_buchis = []
    total_test_mdps = []
    if baseline_type != "eval":
        with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True), name=run_name) as run:
            if method != 'ppo':
                # import pdb; pdb.set_trace()
                sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=reward_type)
                agent, full_orig_crewards, buchi_trajs, mdp_trajs, all_test_crewards, all_test_bvisits, all_test_mdprs = run_Q_continuous(cfg, run, sim, visualize=cfg["visualize"], save_dir=save_dir, agent=None, n_traj=train_trajs)
                total_crewards.extend(full_orig_crewards)
                total_buchis.extend(buchi_trajs)
                total_mdps.extend(mdp_trajs)
                total_test_crewards.extend(all_test_crewards)
                total_test_buchis.extend(all_test_bvisits)
                total_test_mdps.extend(all_test_mdprs)
            else:
                sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=reward_type)
                agent, full_orig_crewards, buchi_trajs, mdp_trajs, all_test_crewards, all_test_bvisits, all_test_mdprs = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=to_hallucinate, visualize=cfg["visualize"],
                                                                save_dir=save_dir, save_model=True, agent=None, n_traj=train_trajs)
                total_crewards.extend(full_orig_crewards)
                total_buchis.extend(buchi_trajs)
                total_mdps.extend(mdp_trajs)
                total_test_crewards.extend(all_test_crewards)
                total_test_buchis.extend(all_test_bvisits)
                total_test_mdps.extend(all_test_mdprs)
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
        test_bvisits, test_mdprew, buchi_visits, mdp_reward, combined_rewards = eval_q_agent(cfg, sim, agent, save_dir=traj_dir)
    else:
        test_bvisits, test_mdprew, buchi_visits, mdp_reward, combined_rewards = eval_agent(cfg, sim, agent, save_dir=traj_dir)
    return total_crewards, total_buchis, total_mdps, total_test_crewards, total_test_buchis, total_test_mdps, (test_bvisits, test_mdprew, buchi_visits, mdp_reward, combined_rewards)
    

if __name__ == "__main__":
    main()