import os
import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam
from policies.sac import GaussianPolicy, QNetwork, DeterministicPolicy, Buffer, hard_update, soft_update
from utls.utls import parse_stl_into_tree


import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
import time
import wandb
from tqdm import tqdm
################################## set device ##################################
# print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
# torch.default_device(device)


class SAC(object):
    def __init__(self, env_space, action_space, param):

        self.gamma = param['gamma']
        self.tau = param['sac']['tau']
        self.alpha = param['sac']['alpha']
        self.ltl_lambda = param['lambda']

        self.policy_type = param['sac']['policy_type']
        self.target_update_interval = param['sac']['iterations_per_target_update']
        self.automatic_entropy_tuning = False

        lr_actor = param['sac']['lr_actor']
        lr_critic = param['sac']['lr_critic']
        self.device = device

        self.critic = QNetwork(env_space, action_space, param).to(device=self.device)


        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.critic_target = QNetwork(env_space, action_space, param).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        self.policy = GaussianPolicy(env_space, action_space, param).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

    def select_action(self, state, buchi):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        #if evaluate is False:
        action, log_prob, mean, is_eps = self.policy.sample(state, buchi)
        if not is_eps:
            return action.detach().cpu().numpy(), log_prob, is_eps
        return action, log_prob, is_eps

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, buchi_batch, action_batch, reward_batch, ltl_reward_batch, constrained_reward_batch, next_state_batch, next_buchi_batch, terminal_batch = memory.sample(batchsize=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        buchi_batch = torch.FloatTensor(buchi_batch).to(self.device).unsqueeze(1).unsqueeze(1).long()
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_buchi_batch = torch.FloatTensor(next_buchi_batch).to(self.device).unsqueeze(1).unsqueeze(1).long()
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        constrained_reward_batch = torch.FloatTensor(constrained_reward_batch).to(self.device)
        terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, is_eps = self.policy.sample(next_state_batch, next_buchi_batch)
            #import pdb; pdb.set_trace()
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_buchi_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.squeeze()
            next_q_value = constrained_reward_batch  + terminal_batch.squeeze() * self.gamma * (min_qf_next_target)
        #import pdb; pdb.set_trace()
        qf1, qf2 = self.critic(state_batch, buchi_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        norm_qf_loss = qf_loss #/ (self.ltl_lambda / (1 - self.gamma))
        #import pdb; pdb.set_trace()
        self.critic_optim.zero_grad()
        norm_qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, is_eps = self.policy.sample(state_batch, buchi_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, buchi_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return norm_qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

def rollout(env, agent, memory, param, i_episode, runner, testing=False, visualize=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    mdp_ep_reward = 0
    ltl_ep_reward = 0
    memory.restart_traj()
    # if testing & visualize:
    #     s = torch.tensor(state['mdp']).type(torch.float)
    #     b = torch.tensor([state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
    #     print(0, state['mdp'], state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
    #     print(agent.Q(s, b))
    
    for t in range(1, param['sac']['T']):  # Don't infinite loop while learning
        action, log_prob, is_eps = agent.select_action(state['mdp'], state['buchi'])
        
        next_state, mdp_reward, done, info = env.step(action, is_eps)
        reward = info['is_accepting']
        rhos = info['rho']
        terminal = info['is_rejecting']
        constrained_reward, _, rew_info = env.constrained_reward(rhos, terminal, state['buchi'], next_state['buchi'], mdp_reward)
        mdp_ep_reward += rew_info["mdp_reward"]
        ltl_ep_reward += rew_info["ltl_reward"]
        if testing & visualize:
            s = torch.tensor(next_state['mdp']).type(torch.float)
            b = torch.tensor([next_state['buchi']]).type(torch.int64).unsqueeze(1).unsqueeze(1)
            print(t, next_state['mdp'], next_state['buchi'], agent.Q(s, b).argmax().to_tensor(0).numpy())
            print(agent.Q(s, b))

        if not testing: # TRAIN ONLY
            # Simulate step for each buchi state
            if not is_eps:
                for buchi_state in range(env.observation_space['buchi'].n):
                    next_buchi_state, is_accepting = env.next_buchi(next_state['mdp'], buchi_state)
                    constrained_reward, _, rew_info = env.constrained_reward(rhos, terminal, state['buchi'], next_state['buchi'], mdp_reward)
                    memory.add(state['mdp'], buchi_state, action, mdp_reward, rew_info['ltl_reward'], constrained_reward, next_state['mdp'], next_buchi_state, is_accepting == -1)
                    if buchi_state == state['buchi']:
                        reward = is_accepting
                        memory.mark()
                
                    # also add epsilon transition 
                    try:                        
                        for eps_idx in range(env.action_space[buchi_state].n):
                            next_buchi_state, is_accepting = env.next_buchi(state['mdp'], buchi_state, eps_idx)
                            constrained_reward, _, rew_info = env.constrained_reward(rhos, terminal, state['buchi'], next_state['buchi'], mdp_reward)
                            memory.add(state['mdp'], buchi_state, action, mdp_reward, rew_info['ltl_reward'], constrained_reward, next_state['mdp'], next_buchi_state, is_accepting == -1)
                    except:
                        pass

            else:
                # no reward for epsilon transition !
                memory.add(state['mdp'], buchi_state, action, mdp_reward, rew_info['ltl_reward'], constrained_reward, next_state['mdp'], next_buchi_state, terminal)
                agent.buffer.mark()
        states.append(next_state['mdp'])
        buchis.append(next_state['buchi'])
        if done:
            break
        state = next_state

    if visualize:
        # frames = 
        # runner.log({"video": wandb.Video([env.render(states=np.atleast_2d(state), save_dir=None) for state in states], fps=10)})
        if testing: 
            runner.log({"testing": wandb.Image(env.render(states=states, save_dir=None))})
        else:
            runner.log({"training": wandb.Image(env.render(states=states, save_dir=None))})
        # agent.buffer.atomics.append(info['signal'])
        # ep_reward += reward
        # disc_ep_reward += param['gamma']**(t-1) * reward

    return mdp_reward, ltl_ep_reward, t

def run_sac(param, runner, env):
    # varphi = mtl.parse(param['ltl']['formula'])
    agent = SAC(env.observation_space, env.action_space, param)
    memory = Buffer(env.observation_space['mdp'].shape, env.action_space['mdp'].shape, max_=param['replay_buffer_size'])
    fixed_state, _ = env.reset()
    checkpoint_count = 0
    num_updates = 0

    for i_episode in tqdm(range(param['q_learning']['n_traj'])):
        # TRAINING
        mdp_reward, ltl_reward, t = rollout(env, agent, memory, param, i_episode, runner, testing=False, visualize=((i_episode % 50) == 0))


        if i_episode % param['q_learning']['update_freq__n_episodes'] == 0:
            num_updates += 1
            qf_loss, policy_loss, _, _  = agent.update_parameters(memory, param['sac']['batch_size'], num_updates)
            # all_losses.append(current_loss)
        
        # if i_episode % param['q_learning']['temp_decay_freq__n_episodes'] == 0:
        #     agent.decay_temp(param['q_learning']['temp_decay_rate'], param['q_learning']['min_action_temp'], param['q_learning']['temp_decay_type'])
        
        # if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
        #     test_data = []
        #     for test_iter in range(param['testing']['num_rollouts']):
        #         test_data.append(rollout(env, agent, memory, param, test_iter, runner, testing=True, visualize= ((i_episode % 50) == 0) & (test_iter == 0) ))
        #     test_data = np.array(test_data)

        # if i_episode % param['checkpoint_freq__n_episodes'] == 0:
        #     ckpt_path = Path(runner.dir) / f'checkpoint_{checkpoint_count}.pt'
        #     torch.save(agent.state_dict(), ckpt_path)
        #     artifact = wandb.Artifact('checkpoint', type='model')
        #     artifact.add_file(ckpt_path)
        #     runner.log_artifact(artifact)
        #     checkpoint_count += 1
    
        if i_episode > 1 and i_episode % 1 == 0:
            avg_timesteps = t #np.mean(timesteps)
            
            # logger.dumpkvs()
            runner.log({
                    'R_LTL': ltl_reward,
                    'R_MDP': mdp_reward,
                    'QF_Loss': qf_loss,
                    'Policy_Loss': policy_loss,
                    #  'TimestepsAlive': avg_timesteps,
                    #  'PercTimeAlive': (avg_timesteps + 1) / param['q_learning']['T'],
                     #'ActionTemp': agent.temp,
                     })
            
