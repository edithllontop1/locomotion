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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 adaptive_weight_teacher = 0,
                 adaptive_weight_alpha=0,
                 adaptive_weight_beta=0,
                 start_teacher = 0.8,
                 start_alpha = 0.7,
                 start_beta = 0.7,
                 end_teacher = 0.5,
                 end_alpha = 0.1,
                 end_beta = 0.1,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.dog = False
        self.prior = False
        self.prior_policy = None
        self.reach_policy = None
        self.teacher_policy = None
        self.adaptive_weight_teacher = adaptive_weight_teacher
        self.adaptive_weight_alpha = adaptive_weight_alpha
        self.adaptive_weight_beta = adaptive_weight_beta
        self.teacher = start_teacher
        self.alpha = start_alpha
        self.beta = start_beta
        self.loss_weight = 1
        self.end_alpha = end_alpha
        self.end_beta = end_beta
        self.end_teacher = end_teacher
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_prior_loss = 0
        mean_reach_loss = 0
        mean_teacher_loss = 0
        self.loss_weight = 1
        prior_loss = 0
        reach_loss = 0
        teacher_loss = 0
        
        if self.prior and self.reach_policy is not None:
            self.loss_weight += 1
        
        if self.prior:
            self.loss_weight -= self.alpha
            if self.adaptive_weight_alpha>0:
                self.alpha *= self.adaptive_weight_alpha
                self.alpha = max(self.alpha,self.end_alpha)
        
        if self.reach_policy is not None:
            self.loss_weight -= self.beta
            if self.adaptive_weight_beta>0:
                self.beta *= self.adaptive_weight_beta
                self.beta = max(self.beta,self.end_beta)
                
        if self.teacher_policy is not None:
            self.loss_weight -= self.teacher
            if self.adaptive_weight_teacher>0:
                self.teacher *= self.adaptive_weight_teacher
                self.teacher = max(self.teacher,self.end_teacher)
            
         
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch,prior_obs_batch, reach_obs_batch, teacher_obs_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                if self.prior:
                    if self.actor_critic.use_RMA:
                        cur_action = self.actor_critic.RMA_actor(obs_batch)[:,:12]
                    else:
                        cur_action = self.actor_critic.actor(obs_batch)[:,:12]
                        
                    if self.dog:
                        prior_action = self.prior_policy.act_inference(prior_obs_batch)[:,:12]
                    else:
                        prior_action = self.prior_policy(prior_obs_batch)#in sim order already
                    prior_loss = (cur_action-prior_action).pow(2).mean()
                    
                    prior_loss = self.alpha*prior_loss
                    
                    # loss = (1-self.alpha)*loss + 
                    # loss = prior_loss
                
                if self.reach_policy is not None:
                    if self.actor_critic.use_RMA:
                        cur_action = self.actor_critic.RMA_actor(obs_batch)[:,12:19]
                    else:
                        cur_action = self.actor_critic.actor(obs_batch)[:,12:19]
                    prior_action = self.reach_policy(reach_obs_batch)
                    if prior_action.shape[1]>7:
                        prior_action = prior_action[:,12:19]
                    reach_loss = (cur_action-prior_action).pow(2).mean()

                    reach_loss = self.beta*reach_loss
                    
                    # loss = (1-self.beta)*loss + self.beta*reach_loss
                    
                if self.teacher_policy is not None: 
                    if self.actor_critic.use_RMA:  
                        cur_action = self.actor_critic.RMA_actor(obs_batch)
                    else:
                        cur_action = self.actor_critic.actor(obs_batch)
                    with torch.no_grad():
                        prior_action = self.teacher_policy.act_inference(teacher_obs_batch)
                    # print((obs_batch-teacher_obs_batch).pow(2).mean())
                    teacher_loss = (cur_action-prior_action).pow(2).mean()

                    teacher_loss = self.teacher*teacher_loss
                
                loss = self.loss_weight*loss + prior_loss + reach_loss + teacher_loss
                    
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if self.prior:
                    mean_prior_loss += prior_loss.item()
                if self.reach_policy is not None:
                    mean_reach_loss += reach_loss.item()
                if self.teacher_policy is not None:
                    mean_teacher_loss += teacher_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_prior_loss /= num_updates
        mean_reach_loss /= num_updates
        mean_teacher_loss /= num_updates
        
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_prior_loss, mean_reach_loss, mean_teacher_loss
