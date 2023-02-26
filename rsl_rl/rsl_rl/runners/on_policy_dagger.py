# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

class OnPolicyDagger:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        self.optimizer = optim.Adam(self.alg.actor_critic.estimator.parameters(), lr=1e-3)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.Loss = nn.MSELoss()

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False, early_stop=-1,prior_policy=None,reach_policy=None, teacher_policy=None):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
     

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(0, tot_iter):
            start = time.time()
            self.env.iter_num += 1
            print("env.iter",self.env.iter_num,self.alg.alpha,self.alg.beta,self.alg.teacher,self.alg.loss_weight)
            # Rollout
            h = torch.zeros((self.alg.actor_critic.estimator.num_layers,self.env.num_envs,self.alg.actor_critic.estimator.hidden_size),device=self.device, requires_grad=True)
            c = torch.zeros((self.alg.actor_critic.estimator.num_layers,self.env.num_envs,self.alg.actor_critic.estimator.hidden_size),device=self.device, requires_grad=True)
            
            hidden_state = (h,c)
            mask = torch.ones((self.env.num_envs,),device=self.device)
            
            loss = 0
            obs_hist = [torch.zeros_like(obs) for _ in range(self.alg.actor_critic.estimator_hist_len)]
            for _ in range(self.num_steps_per_env):  
                # hidden_state = hidden_state * masks
                # z, hidden_state = actor_critic.get_student_latent() # with gradient -- might require to add an extra time dimension (unsqueeze(0))
                hidden_state = (torch.einsum("ijk,j->ijk",hidden_state[0],mask),   
                                torch.einsum("ijk,j->ijk",hidden_state[1],mask))
                
                obs_hist.pop(0);obs_hist.append(obs);estimator_obs=torch.cat(obs_hist,dim=-1)
                zhat,hidden_state = self.alg.actor_critic.estimate_latent(estimator_obs.unsqueeze(0),hidden_state)  
                # z_expert = actor_critic.get_expert_latent(obs) # with torch.no_grad()
                with torch.no_grad():
                    zexpert = self.alg.actor_critic.acquire_latent(obs)      
                
                # loss += mse(z, z_expert)
                loss +=self.Loss(zhat[0], zexpert)
                # actions = actor_critic.get_student_action_inference(z.detach()) # with torch.no_grad()
                with torch.no_grad():
                    actions = self.alg.actor_critic.estimate_actor(obs,zhat[0].detach())
             
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                
                mask = ~dones #mask out finished envs
                
                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            
            loss = loss / self.num_steps_per_env
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
            stop = time.time()
            learn_time = stop - start
            
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

            if early_stop >0 and it>early_stop:
                if statistics.mean(rewbuffer) <= 1e-5:
                    self.current_learning_iteration += num_learning_iterations
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
                    return
        self.scheduler.step()
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        return

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/BSloss', locs['loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'BC loss:':>{pad}} {locs['loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'BC loss:':>{pad}} {locs['loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            "estimator": self.alg.actor_critic.estimator.state_dict(),
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True, reset_std=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'],strict=False)
       
        example_input = torch.rand(1, self.alg.actor_critic.actor_obs_dim)
        print("self.alg.actor_critic.actor_obs_dim",self.alg.actor_critic.actor_obs_dim)
        actor_graph = torch.jit.trace(self.alg.actor_critic.actor.to('cpu'), example_input)
        torch.jit.save(actor_graph, "/home/ravenhuang/amp/real_policy/0205/actor.pt")
        
        example_input = torch.rand(1, self.alg.actor_critic.estimator_obs_dim)
        print("self.alg.actor_critic.estimator_obs_dim",self.alg.actor_critic.estimator_obs_dim)
        h = torch.zeros((self.alg.actor_critic.estimator.num_layers,1,self.alg.actor_critic.estimator.hidden_size))
        c = torch.zeros((self.alg.actor_critic.estimator.num_layers,1,self.alg.actor_critic.estimator.hidden_size))
        
        actor_graph = torch.jit.trace(self.alg.actor_critic.estimator.to('cpu'), (example_input[None,...],h,c))
        torch.jit.save(actor_graph, "/home/ravenhuang/amp/real_policy/0205/estimator.pt")
        
        if reset_std:
            self.alg.actor_critic.std = nn.Parameter(self.alg.actor_critic.init_noise_std * torch.ones(self.alg.actor_critic.num_actions,device=self.device))
        
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_jit(self,path):
        loaded_policy = torch.jit.load(path)
        return loaded_policy

    def get_inference_policy(self, device=None, sample=False):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        if sample:
            return self.alg.actor_critic.act
        else:
            return self.alg.actor_critic.act_inference