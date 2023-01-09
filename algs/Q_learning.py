import torch

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np

from algs.Q_discrete import Q_learning as Q_learning_discrete
from algs.Q_continuous import Q_learning as Q_learning_continuous

def rollout(env, agent, param, i_episode, testing=False, visualize=False):
    state, _ = env.reset()
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    # if testing & visualize:
    #     s = torch.tensor(state['mdp']).type(torch.float)
    #     b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
    #     print(0, state['mdp'], state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
    #     print(agent.Q(s, b))
    
    for t in range(1, param['T']):  # Don't infinite loop while learning
        action, is_eps, log_prob = agent.select_action(state, testing)
        
        next_state, cost, done, info = env.step(action, is_eps)
        reward = info['is_accepting']

        # if testing & visualize:
        #     s = torch.tensor(next_state['mdp']).type(torch.float)
        #     b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
        #     print(t, next_state['mdp'], next_state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
        #     print(agent.Q(s, b))

        if not testing: # TRAIN ONLY
            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, is_accepting = env.next_buchi(next_state['mdp'], buchi_state)
                    agent.collect(state['mdp'], buchi_state, action, is_accepting, next_state['mdp'], next_buchi_state)
                    if buchi_state == state['buchi']:
                        reward = is_accepting
                        agent.buffer.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            eps_action = env.action_space['mdp'].n + eps_idx
                            next_buchi_state, is_accepting = env.next_buchi(state['mdp'], buchi_state, eps_action)
                            agent.collect(state['mdp'], buchi_state, eps_action, is_accepting, state['mdp'], next_buchi_state)
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], state['buchi'], action, 0, next_state['mdp'], next_state['buchi'])
                agent.buffer.mark()
                reward = 0

        if visualize:
            env.render()
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        if done:
            break
        state = next_state

    return ep_reward, disc_ep_reward, t
        
def run_Q_learning(param, env, second_order = False, continuous_state=True):
    
    if continuous_state:
        agent = Q_learning_continuous(env.observation_space, env.action_space, param['gamma'], param)
    else:
        agent = Q_learning_discrete(env.observation_space, env.action_space, param['gamma'], param)
    fig, axes = plt.subplots(2, 1)
    history = []
    success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()

    for i_episode in range(param['n_traj']):
        # TRAINING
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, testing=False)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, test_iter, testing=True, visualize= ((i_episode % 50) == 0) & (test_iter == 0) ))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            history += [disc_ep_reward]
            # success_history += [env.did_succeed(state, action, reward, next_state, done, t + 1)]
            success_history += [test_data[:, 0].mean()]
            method = 'TR' if second_order else 'Adam'
            plot_something_live(axes, [np.arange(len(history)),  np.arange(len(success_history))], [history, success_history], method)
            logger.logkv('Iteration', i_episode)
            logger.logkv('Method', method)
            logger.logkv('Success', success_history[-1])
            logger.logkv('Last20Success', np.mean(np.array(success_history[-20:])))
            logger.logkv('EpisodeReward', ep_reward)
            logger.logkv('DiscEpisodeReward', disc_ep_reward)
            logger.logkv('TimestepsAlive', avg_timesteps)
            logger.logkv('PercTimeAlive', (avg_timesteps+1)/param['T'])
            logger.logkv('ActionTemp', agent.temp)
            
            logger.dumpkvs()
            
    plt.close()