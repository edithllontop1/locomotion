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
class A1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_hist_obs = 5
        num_observations = 41*num_hist_obs
        num_actions = 12
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    
    class commands:
        curriculum = False
        discrete = ["lin_vel_x"]
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        active = 0.35
        class ranges:
            lin_vel_x = [0., 0.35] # min max [m/s] 0.2/0.8
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0] #or [-pi,pi] when yaw=0
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5*torch.ones(12)
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        terminate_condition = {}
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        
        exponential_decay = []
        max_iter = 8001
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            feet_air_time =  0.0
            termination = -3.0
            tracking_lin_vel = 7.0
            tracking_ang_vel = -2
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.01
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            collision = -0.01
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            total_work = -0.001
            action_norm = -0.001
        
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
        curriculum = ["friction"]
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 1
        min_push_vel_xy = 50.
        max_push_vel_xy = 300.
                    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # horizontal_scale = 0.05 # [m]
        # vertical_scale = 0.005 # [m]
        # border_size = 25 # [m]
        # curriculum = False
        # static_friction = 1.
        # dynamic_friction = 1.
        # restitution = 0.
        # # rough terrain only:
        # rough_plane = True
        # roughness = 0.05
        
        # slope = [-0.7,0.5]
        # slope_portion = 0. #this determines how many slopes to generate
        # slope_roughness = 0.
        
        # rough_stair = False
        # stair_size = [1.5,0.35]
        
        # measure_heights = True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # selected = False # select a unique terrain type and pass all arguments
        # terrain_kwargs = None # Dict of arguments for selected terrain
        # max_init_terrain_level = 5 # starting curriculum state
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows= 10 # number of terrain rows (levels)
        # num_cols = 20 # number of terrain cols (types)
        # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0., 1.0 , 0, 0, 0]
        # # trimesh only:
        # slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        
class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'tanh'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24
        load_run = "Dec20_21-49-16_walk_still_delay"
        resume = True
        reset_std = True
        checkpoint = -1
        max_iterations = 8001 # number of policy updates
        experiment_name = 'a1'
        run_name = 'walk_still_delay'
        


  