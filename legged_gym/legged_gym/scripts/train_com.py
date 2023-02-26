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
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR



class Expert:
    def __init__(self, policy, device='cpu', baseDim=46):
        self.policy = policy
        self.policy.to(device)
        for params in self.policy.parameters():
            params.requires_grad = False
        self.device = device
        self.baseDim = baseDim 
        # 19 is size of priv info for the cvpr policy
        self.end_idx = baseDim + 19 #-self.geomDim*(self.n_futures+1) - 1
        
    def reorder(self, inp):
        reorder_indices = torch.tensor([3,4,5,0,1,2,9,10,11,6,7,8],device=self.device).long()
        return torch.index_select(inp,1,reorder_indices)
        # return torch.index_select(inp.cpu(),1,torch.LongTensor(reroder_indices)).to(self.device)

    def __call__(self, obs):
        with torch.no_grad():
            resized_obs = obs[:, :self.end_idx]
            latent = self.policy.info_encoder(resized_obs[:,self.baseDim:])
            input_t = torch.cat((resized_obs[:,:self.baseDim], latent), dim=1)
            action = self.policy.action_mlp(input_t)
            #make the policy output in sim order
            return self.reorder(action)
        
        
# negative y force: waist 1.57 shoulder -1.88 elbow -1.88
# positive y force -1.57 -1.88 elbow -1.88
# positive x force -2.35
# nagative x force -0.78

class armPolicy:
    def __init__(self,device,action_scale,impulse):
        self.device = device
        self.action_scale = action_scale
        self.py_px = torch.tensor([1.57,0,-3.1,0,0,0,0],device=self.device)/self.action_scale
        self.py_nx = torch.tensor([1.57,0,-3.1,0,0,0,0],device=self.device)/self.action_scale
        self.ny_px = torch.tensor([-1.57,0,-3.1,0,0,0,0],device=self.device)/self.action_scale
        self.ny_nx = torch.tensor([-1.57,0,-3.1,0,0,0,0],device=self.device)/self.action_scale
        
        if impulse:
            self.py_px = -self.py_px
            self.py_nx = -self.py_nx
            self.ny_px = -self.ny_px
            self.ny_nx = -self.ny_nx
        

    def __call__(self, obs):
        if obs[0,0]<1:
            return  torch.zeros((obs.shape[0],7),device=self.device)
        else:
            force_signs = obs[:,1:3]
            actions = torch.zeros((obs.shape[0],7),device=self.device)
            py_px_indice = torch.where(torch.logical_and(force_signs[:,0]>0,force_signs[:,1]>0))[0]
            py_nx_indice = torch.where(torch.logical_and(force_signs[:,0]<0,force_signs[:,1]>0))[0]
            ny_px_indice = torch.where(torch.logical_and(force_signs[:,0]>0,force_signs[:,1]<0))[0]
            ny_nx_indice = torch.where(torch.logical_and(force_signs[:,0]<0,force_signs[:,1]<0))[0]
            # if len(py_px_indice)>0 or len(py_nx_indice)>0 or len(ny_px_indice)>0 or len(ny_nx_indice)>0:
            #     import pdb;pdb.set_trace()
            actions[py_px_indice,:] = self.py_px
            actions[py_nx_indice,:] = self.py_nx
            actions[ny_px_indice,:] = self.ny_px
            actions[ny_nx_indice,:] = self.ny_nx
            return actions
            
    
def train(args):
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    log_dir = ppo_runner.log_dir
    os.makedirs(log_dir,exist_ok=True)
    shutil.copy2(f'./envs/{args.task}/{args.task}_config.py',log_dir+f'/{args.task}_config_log.py')
    
    path = "/home/ravenhuang/amp/locomani/flat_ckpt/0000/policy_16800.pt"
    graph = torch.jit.load(path,map_location=env.device)
    policy = Expert(graph,env.device)
    
    
    reach_policy = None    
    if getattr(env.cfg.env,"reach",False):
        reach_policy = armPolicy(env.device,env.cfg.control.action_scale[-1],env.cfg.domain_rand.impulse)
        # reach_path = "/home/ravenhuang/amp/locomani/legged_gym/logs/a1wx/exported/policies/policy_he.pt"
        # reach_policy =  torch.jit.load(reach_path,map_location=env.device)
    
    teacher_policy =None
    if getattr(env.cfg.env,"teacher",False):
        path = env.cfg.env.teacher_policy_path
        teacher_policy = torch.jit.load(path,map_location=env.device)
        for params in teacher_policy.parameters():
            params.requires_grad = False
        
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True,prior_policy=policy,reach_policy=reach_policy,teacher_policy=teacher_policy)

if __name__ == '__main__':
    args = get_args()
    train(args)
