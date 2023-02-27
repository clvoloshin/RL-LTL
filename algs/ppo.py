import torch

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np

from algs.ppo_discrete import PPO as PPO_discrete
from algs.ppo_continuous import PPO as PPO_continuous
from tqdm import tqdm

def rollout(env, agent, param, i_episode, testing=False, visualize=False, to_hallucinate=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    
    for t in range(1, param['ppo']['T']):  # Don't infinite loop while learning
        # tic = time.time()
        action, action_idx, is_eps, log_prob, all_logprobs = agent.select_action(state, testing)
        # total_action_time += time.time() - tic 

        if is_eps:
            action = int(action)
        else:
            action = action.cpu().numpy().flatten()
        
        try:
            next_state, cost, done, info = env.step(action, is_eps)
        except:
            next_state, cost, done, _, info = env.step(action, is_eps)
        reward = int(info['is_accepting'])
        if testing & visualize:
            print(next_state['mdp'])
            print(next_state['buchi'])
            try:
                print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
            except:
                pass
            print(action)
            # print(agent.Q(s, b))

        # tic = time.time()
        agent.buffer.add_experience(env, state['mdp'], state['buchi'], action, info['is_accepting'], next_state['mdp'], next_state['buchi'], action_idx, is_eps, all_logprobs)
        # total_experience_time += time.time() - tic

        # if visualize:
        #     env.render()
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        states.append(next_state['mdp'])
        buchis.append(next_state['buchi'])
        if done:
            break
        state = next_state

    if visualize:
        try:
            env.render(states=env.unnormalize(states), save_dir=logger.get_dir() + '/' + "episode_" + str(i_episode))
        except:
            pass

    # print('Get Experience', total_experience_time)
    # print('Get Action', total_action_time)
    print(next_state['mdp'])
    print(next_state['buchi'])
    try:
        print(env.mdp.distances_to_wp(next_state['mdp'][0], next_state['mdp'][1])[1:11])
    except:
        pass
    print(action)
    return ep_reward, disc_ep_reward, t
        
def run_PPO(param, env, second_order = False, continuous_state=True, to_hallucinate=False):
    
    if continuous_state:
        agent = PPO_continuous(env.observation_space, env.action_space, param['gamma'], param, to_hallucinate)
    else:
        agent = PPO_discrete(env.observation_space, env.action_space, param['gamma'], param, to_hallucinate)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    disc_success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in tqdm(range(param['ppo']['n_traj'])):
        # TRAINING

        
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, testing=False)
        agent.update()
        
        if i_episode % param['ppo']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['ppo']['temp_decay_rate'], param['ppo']['min_action_temp'], param['ppo']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, i_episode, testing=True, visualize= test_iter == 0 )) #param['n_traj']-100) ))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [disc_ep_reward]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            success_history += [test_data[:, 0].mean()]
            disc_success_history += [test_data[:, 1].mean()]
            method = 'Ours' if to_hallucinate else 'Baseline'
            plot_something_live(axes, [np.arange(len(history)),  np.arange(len(success_history))], [history, success_history], method)
            logger.logkv('Iteration', i_episode)
            logger.logkv('Method', method)
            logger.logkv('Success', success_history[-1])
            logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            logger.logkv('DiscSuccess', disc_success_history[-1])
            logger.logkv('Last20DiscSuccess', np.mean(np.array(disc_success_history[-20:])))
            logger.logkv('EpisodeReward', ep_reward)
            logger.logkv('DiscEpisodeReward', disc_ep_reward)
            logger.logkv('TimestepsAlive', avg_timesteps)
            logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['ppo']['T'])
            logger.logkv('ActionTemp', agent.temp)
            logger.dumpkvs()
            
    plt.close()