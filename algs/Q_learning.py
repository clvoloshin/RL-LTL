import torch

import logger
from utls.plotutils import plot_something_live
import numpy as np
import wandb
from algs.Q_discrete import Q_learning as Q_learning_discrete
from algs.Q_continuous import Q_learning as Q_learning_continuous

from tqdm import tqdm

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False, to_hallucinate=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    ep_reward = 0
    disc_ep_reward = 0
    if not testing: agent.buffer.restart_traj()
    # if testing & visualize:
    #     s = torch.tensor(state['mdp']).type(torch.float)
    #     b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
    #     print(0, state['mdp'], state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
    #     print(agent.Q(s, b))
    
    for t in range(1, param['q_learning']['T']):  # Don't infinite loop while learning
        action, is_eps, log_prob = agent.select_action(state, testing)
        
        next_state, cost, done, info = env.step(action, is_eps)
        reward = info['is_accepting']

        # if testing & visualize:
        #     s = torch.tensor(next_state['mdp']).type(torch.float)
        #     b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
        #     print(t, next_state['mdp'], next_state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
        #     print(agent.Q(s, b))

        if (not testing) and to_hallucinate: # TRAIN ONLY
            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, accepting_rejecting_neutal = env.next_buchi(next_state['mdp'], buchi_state)
                    is_accepting = int(accepting_rejecting_neutal > 0)
                    agent.collect(state['mdp'], buchi_state, action, is_accepting, next_state['mdp'], next_buchi_state)
                    if buchi_state == state['buchi']:
                        reward = is_accepting
                        agent.buffer.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            eps_action = env.action_space['mdp'].n + eps_idx
                            next_buchi_state, accepting_rejecting_neutal = env.next_buchi(state['mdp'], buchi_state, eps_idx)
                            is_accepting = int(accepting_rejecting_neutal > 0)
                            agent.collect(state['mdp'], buchi_state, eps_action, is_accepting, state['mdp'], next_buchi_state)
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], state['buchi'], action, 0, next_state['mdp'], next_state['buchi'])
                agent.buffer.mark()
                reward = 0
        elif (not testing):
            if not is_eps:
                agent.collect(state['mdp'], state['buchi'], action, reward, next_state['mdp'], next_state['buchi'])
                agent.buffer.mark()
            else:
                # no reward for epsilon transition !
                agent.collect(state['mdp'], state['buchi'], action, 0, next_state['mdp'], next_state['buchi'])
                agent.buffer.mark()
                reward = 0


        
        # agent.buffer.atomics.append(info['signal'])
        ep_reward += reward
        disc_ep_reward += param['gamma']**(t-1) * reward

        states.append(next_state['mdp'])
        buchis.append(next_state['buchi'])
        if done:
            break
        state = next_state
    
    if visualize:
        # frames = 
        # runner.log({"video": wandb.Video([env.render(states=np.atleast_2d(state), save_dir=None) for state in states], fps=10)})
        if testing: 
            img = env.render(states=states, save_dir=None)
            runner.log({"testing": wandb.Image(img)})
        else:
            runner.log({"training": wandb.Image(env.render(states=states, save_dir=None))})
    return ep_reward, disc_ep_reward, t
        
def run_Q_learning(param, runner, env, second_order = False, continuous_state=True, to_hallucinate=False):
    
    if continuous_state:
        agent = Q_learning_continuous(env.observation_space, env.action_space, param['gamma'], param)
    else:
        agent = Q_learning_discrete(env.observation_space, env.action_space, param['gamma'], param)
    history = []
    success_history = []
    disc_success_history = []
    running_reward = 10
    fixed_state, _ = env.reset()
    current_loss = 1
    for i_episode in tqdm(range(param['q_learning']['n_traj'])):
        # TRAINING
        ep_reward, disc_ep_reward, t = rollout(env, agent, param, i_episode, runner, testing=False, to_hallucinate=to_hallucinate)
        
        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            current_loss = agent.update()
        
        if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_data = []
            for test_iter in range(param['testing']['num_rollouts']):
                test_data.append(rollout(env, agent, param, i_episode, runner, testing=True, visualize= test_iter == 0 ))
            test_data = np.array(test_data)
    
        if i_episode % 1 == 0:
            runner.log({#'Iteration': i_episode,
                    #  'last_reward': last_reward,
                    #  'Method': method,
                    #  'Success': success_history[-1],
                    #  'Last20Success': np.mean(np.array(success_history[-20:])),
                    #  'DiscSuccess': disc_success_history[-1],
                    #  'Last20DiscSuccess': np.mean(np.array(disc_success_history[-20:])),
                    #  'EpisodeReward': ep_reward,
                    #  'DiscEpisodeReward': disc_ep_reward,
                    # 'EpisodeRhos': stl_val,
                    # 'ExpectedQVal': max_q_val,
                    'Reward': ep_reward,
                    'DiscountedReward': disc_ep_reward,
                    'LossVal': current_loss,
                    'AvgTimesteps': t,
                    #  'TimestepsAlive': avg_timesteps,
                    #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                     'ActionTemp': agent.temp,
                     })
            
