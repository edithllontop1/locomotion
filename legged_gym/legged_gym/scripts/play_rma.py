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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path
from collections import deque
import statistics
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_com import armPolicy


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.eval = True
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.env.episode_length_s = 10
    env_cfg.env.vis = True
    env_cfg.domain_rand.curriculum = []
    # env_cfg.rewards.scales.episode_length = 1
    # env_cfg.rewards.scales.x_distance = 1
    # env_cfg.rewards.scales.y_distance = 1
    # env_cfg.domain_rand.push_robots = False

    plot_hist=True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = os.path.join(log_root, train_cfg.runner.load_run)
    
    # path = "/home/ravenhuang/locomani/policywp.pt"
    # graph = torch.jit.load(path,map_location='cpu')
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        path = os.path.join(resume_path, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 50 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    print("stop_rew_log",stop_rew_log)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    died_envs = torch.zeros(env_cfg.env.num_envs, device=env.device, dtype=torch.long)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    lenbuffer = deque(maxlen=600)
    
    h = torch.zeros((ppo_runner.alg.actor_critic.estimator.num_layers,ppo_runner.env.num_envs,ppo_runner.alg.actor_critic.estimator.hidden_size),device=ppo_runner.device, requires_grad=True)
    c = torch.zeros((ppo_runner.alg.actor_critic.estimator.num_layers,ppo_runner.env.num_envs,ppo_runner.alg.actor_critic.estimator.hidden_size),device=ppo_runner.device, requires_grad=True)
    hidden_state = (h,c)
    mask = torch.ones((env.num_envs,),device=env.device)
    obs_hist = [torch.zeros_like(obs) for _ in range(ppo_runner.alg.actor_critic.estimator_hist_len)]
    
    for i in range(10*int(env.max_episode_length)):
        # print(env.base_lin_vel)
        hidden_state = (torch.einsum("ijk,j->ijk",hidden_state[0],mask),   
                                torch.einsum("ijk,j->ijk",hidden_state[1],mask))
        obs_hist.pop(0);obs_hist.append(obs);estimator_obs=torch.concatenate(obs_hist,dim=-1)
        with torch.no_grad():
            zhat,hidden_state = ppo_runner.alg.actor_critic.estimate_latent(estimator_obs.unsqueeze(0),hidden_state)  
        
        actions = ppo_runner.alg.actor_critic.estimate_actor_inference(obs,zhat[0].detach())
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        mask = ~dones
        
        cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        cur_episode_length[new_ids] = 0
        
        died_envs += (dones * ~env.time_out_buf) #record any env that has died once
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        if 0<i<stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale[joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
            
        if  i%stop_rew_log!=0:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i%stop_rew_log==0 and i>0:
            print(f"Mean episode length: {statistics.mean(lenbuffer)}")
            # print(list(lenbuffer))
            succ, results_dict = logger.print_rewards(died_envs)
            del logger
            env.reset()
            died_envs = torch.zeros(env_cfg.env.num_envs, device=env.device, dtype=torch.long)
            logger = Logger(env.dt)

            if plot_hist:
                # out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.load_run, 'episode_length_histograms')
                out_dir = resume_path
                os.makedirs(out_dir, exist_ok=True)

                plt.figure()
                n, bins, patches = plt.hist(x=list(lenbuffer), bins=20, color='#0504aa')
                plt.grid(axis='y', alpha=0.75)
                plt.xlim([0, env.max_episode_length+10])    
                plt.ylim([0, env_cfg.env.num_envs+10])
                plt.xlabel('Episode Length')
                plt.ylabel('Number of Robots')
                plt.title('Episode Lengths')
                
                plt.savefig(os.path.join(out_dir, f"min_{env_cfg.domain_rand.min_push_vel_xy}_max_{env_cfg.domain_rand.max_push_vel_xy}_it_{i}_succ_{succ}.png"))

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
