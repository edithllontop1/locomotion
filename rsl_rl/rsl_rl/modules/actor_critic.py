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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
# from .actor_critic_recurrent import Memory
class RNNEstimator(torch.nn.Module):
    def __init__(self, input_size, output_size, type='lstm', num_layers=1, hidden_size=256, privilege_mask=None ):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        
        layers = [nn.Linear(hidden_size, output_size),nn.Tanh()]
        self.last_mlp = nn.Sequential(*layers)
        
        self.privilege_mask = privilege_mask
    
    # def forward(self,observations, hidden_state):
    #     out, hidden_state = self.rnn(observations[...,~self.privilege_mask],hidden_state)
    #     latent = self.last_mlp(out)
    #     return latent, hidden_state
    def forward(self,observations, h,c):
        hidden_state = (h,c)
        out, hidden_state = self.rnn(observations,hidden_state)
        latent = self.last_mlp(out)
        return latent, hidden_state[0], hidden_state[1]
        
class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        RMA=False,
                        RMA_hidden_dims=[128,128],
                        privilege_obs_dim=0,
                        num_hist_obs=0,
                        num_privilege_obs = 0,
                        num_latent = 8,
                        estimator = False,
                        estimator_hist_len = 1,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        if RMA or estimator:
            mlp_input_dim_a = mlp_input_dim_a - num_privilege_obs + num_latent
        self.actor_obs_dim = mlp_input_dim_a
        self.estimator_obs_dim = num_actor_obs - num_privilege_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        #RMA layers
        self.use_RMA = RMA
        self.use_estimator = estimator
        
        if self.use_RMA or self.use_estimator:
            privilege_mask = torch.zeros((num_hist_obs,num_actor_obs//num_hist_obs))
            privilege_mask[:,-privilege_obs_dim:] = 1
            privilege_mask = privilege_mask.flatten().to(bool)
            self.privilege_mask = privilege_mask
            
        if self.use_RMA:
            RMA_layers = []
            RMA_layers.append(nn.Linear(num_privilege_obs, RMA_hidden_dims[0]))
            RMA_layers.append(activation)
            for l in range(len(RMA_hidden_dims)):
                if l == len(RMA_hidden_dims) - 1:
                    RMA_layers.append(nn.Linear(RMA_hidden_dims[l], num_latent))
                    RMA_layers.append(nn.Tanh())
                else:
                    RMA_layers.append(nn.Linear(RMA_hidden_dims[l], RMA_hidden_dims[l + 1]))
                    RMA_layers.append(activation)
            self.RMA = nn.Sequential(*RMA_layers)
        
        #if learning an estimator
        if self.use_estimator:
            self.estimator_hist_len = estimator_hist_len
            self.estimator_obs_mask = self.privilege_mask.repeat(self.estimator_hist_len)
            self.estimator = RNNEstimator(self.estimator_hist_len*(num_actor_obs- num_privilege_obs), num_latent, type="lstm", num_layers=1, hidden_size=256, privilege_mask=self.privilege_mask)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.init_noise_std = init_noise_std
        self.num_actions = num_actions
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    
    def estimate_latent(self,observations,hidden_states=None):
        latent, h,c =  self.estimator(observations[...,~self.estimator_obs_mask], hidden_states[0], hidden_states[1])
        return latent, (h,c)
    
    def estimate_actor(self,observations,latent):
        new_obs = torch.cat([observations[...,~self.privilege_mask],latent],dim=1).to(observations.device)
        mean = self.actor(new_obs)
        self.distribution = Normal(mean, mean*0. + torch.min(self.std,self.init_noise_std*torch.ones_like(self.std)))
        return self.distribution.sample()
    
    def estimate_actor_inference(self,observations,latent):
        new_obs = torch.cat([observations[...,~self.privilege_mask],latent],dim=1).to(observations.device)
        mean = self.actor(new_obs)
        return mean
    
    # def estimate_actor(self,observations,hidden_states=None):
    #     latent, hidden_states = self.estimate_latent(observations, hidden_states)
    #     new_obs = torch.cat([observations[:,~self.privilege_mask],latent],dim=1).to(observations.device)
    #     return self.actor(new_obs)
    
    def acquire_latent(self, observations):
        priviledge_obs = observations[:,self.privilege_mask]
        return self.RMA(priviledge_obs)
        
    def RMA_actor(self, observations):
        latent = self.acquire_latent(observations)
        new_obs = torch.cat([observations[:,~self.privilege_mask],latent],dim=1).to(observations.device)
        return self.actor(new_obs)

    def update_distribution(self, observations):
        if self.use_RMA:
            mean = self.RMA_actor(observations)
        else:
            mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + torch.min(self.std,self.init_noise_std*torch.ones_like(self.std)))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.use_RMA:
            actions_mean = self.RMA_actor(observations) 
        else:
            actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
