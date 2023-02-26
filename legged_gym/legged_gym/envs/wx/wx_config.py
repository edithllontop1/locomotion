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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch
class WXRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_hist_obs = 1
        num_observations = 24*num_hist_obs
        num_actions = 7
        episode_length_s = 3 # episode length in seconds
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            
            'waist':0,
            'shoulder':-1.88,
            'elbow':1.5,
            'wrist_angle':0.8,
            'wrist_rotate':0,
            'left_finger':0,
            'right_finger':0
            
        }
    
    class commands:
        num_commands = 3  # default: xyz ee pos
        resampling_time = 3. # time before command are changed[s]
        class ranges:
            x = [-0.2, 0.2] 
            y = [-0.2, 0.2] 
            z = [0.05, 0.25] 

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'waist': 100.,'shoulder': 300.,'elbow': 250.,'wrist_angle': 50.,'wrist_rotate': 30.,'left_finger': 50.,'right_finger': 50.}  # [N*m/rad]
        damping = {'waist': 0.,'shoulder': 0.,'elbow': 0.,'wrist_angle': 0.,'wrist_rotate': 1.,'left_finger': 5.,'right_finger': 5.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.5*torch.ones(7)
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wx/urdf/wx200.urdf'
        name = "wx"
        foot_name = "finger"
        penalize_contacts_on = ["wx200/forearm_link","wx200/wrist_link","wx200/gripper_link","wx200/left_finger_link", "wx200/right_finger_link"]
        terminate_after_contacts_on = []
        terminate_condition = {}
        fix_base_link = True
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards:
        soft_dof_pos_limit = 0.99
        exponential_decay = []
        max_iter = 1001
        class scales:
            torques = -0.000
            dof_pos_limits = -0.0
            dist = 1
            action_rate = -0.0
            total_work = -0.0
            action_norm = -0.00
            collision = -0
            succ = 15.
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class noise:
        add_noise = False
        noise_level = 1.0# scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class domain_rand:
        # curriculum = ["friction"]
        curriculum = ["friction","mass"]
        randomize_friction = False
        friction_range = [-0.2, 0.2] #deviation from 1
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 1
        min_push_vel_xy = 50.
        max_push_vel_xy = 300.
                    
            
    class terrain( LeggedRobotCfg.terrain ):        
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False
        
class WXRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # activation = 'tanh'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 32
        # load_run = -1
        load_run = "Dec31_03-17-32_reach"
        resume = False
        reset_std = True
        checkpoint = 700
        max_iterations = 1001 # number of policy updates
        experiment_name = 'wx'
        run_name = 'reach'
        
  