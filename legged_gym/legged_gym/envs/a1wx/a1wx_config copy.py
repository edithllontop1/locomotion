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
class A1WXRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 5200
        num_hist_obs = 2
        # num_observations = 71
        num_observations = (62+3+3+4+1+1)*num_hist_obs
        num_actions = 19
        delay_len = 2
        delay_curriulum = False
        
        num_observations_prior = 65
        num_observations_reach = 5
        
        push_steps = 1
        # vis = True
        # reach = True
        # train_dog=True
        # prior=True
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': 0.05 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            
            'FL_hip_joint': -0.05,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.4,   # [rad]
            
            'RR_hip_joint': 0.05,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.4,    # [rad]
            
            'RL_hip_joint': -0.05,   # [rad]
            'RL_thigh_joint': .8,   # [rad]
            'RL_calf_joint': -1.4,    # [rad]
                   
            'waist':0,
            'shoulder':-1.88,
            'elbow':1.5,
            'wrist_angle':0.,
            'wrist_rotate':0,
            'left_finger':0,
            'right_finger':0
            
        }
    
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            
    class commands:
        curriculum = False
        # discrete = ["lin_vel_x"]
        discrete = []
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        active = 1.0
        class ranges:
            ratio = 0.5 #ratio of sample 0 
            lin_vel_x = [0., 0.3] # min max [m/s] 0.2/0.8
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0] #or [-pi,pi] when yaw=0
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 55.,'waist': 200.,'shoulder': 300.,'elbow': 250.,'wrist_angle': 50.,'wrist_rotate': 30.,'left_finger': 50.,'right_finger': 50.}  # [N*m/rad]
        # damping = {'joint': 0.6,'waist': 0.,'shoulder': 0.,'elbow': 0.,'wrist_angle': 0.,'wrist_rotate': 1.,'left_finger': 5.,'right_finger': 5.}     # [N*m*s/rad]
        stiffness = {'joint': 55.,'waist': 55.,'shoulder': 55.,'elbow': 55.,'wrist_angle': 55.,'wrist_rotate': 55.,'left_finger': 55.,'right_finger': 55.}  # [N*m/rad]
        damping = {'joint': 0.6,'waist': 0.6,'shoulder': 0.6,'elbow': 0.6,'wrist_angle': 0.6,'wrist_rotate': 0.6,'left_finger': 0.6,'right_finger':0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = torch.ones(19)
        action_scale[:-7] = 0.4
        action_scale[-7:] = 3.1
        action_scale[-4:] = 0
        # action_scale[-7:] = 0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1wx/urdf/a1wx.urdf'
        name = "a1wx"
        foot_name = "foot"
        hip_name = 'hip'
        penalize_contacts_on = ["thigh", "calf","wx200/forearm_link","wx200/wrist_link","wx200/gripper_link","wx200/left_finger_link", "wx200/right_finger_link"]
        terminate_after_contacts_on = ["base","wx200/forearm_link","wx200/wrist_link","wx200/gripper_link","wx200/left_finger_link", "wx200/right_finger_link"]   
        # terminate_condition = {'base_height':0.17}
        # terminate_after_contacts_on = ["base","wx200/forearm_link","wx200/wrist_link","wx200/gripper_link"]  
        terminate_condition = {}
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True
        
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.97
        base_height_target = 0.25
        # exponential_decay = ["tracking_ang_vel","total_work","action_norm"]
        exponential_decay = []
        max_iter = 1000
        arm_weight = [2, 1.5, 1.6, 1, 1, 1,1]
        work_weight = [1.2, 1., 1., 1, 1, 1,1]
        
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000
            dof_pos_limits = -0.0
            feet_air_time =  0.0
            termination = -5.0
            tracking_lin_vel = 7.0
            tracking_ang_vel = -0.5
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = 1.
            dof_vel = -0.
            dof_acc = -0
            # base_height = -3
            # collision = -10.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dog_work = -0.0021
            arm_work = -0.008
            dog_action_norm = -0.001
            arm_action_norm = -0.001
            # base_vel = -0.05
            # roll_pitch = 20.
            # arm_motion = 0.1
            # hip_norm = -2
            inward = -0.1
            lin_vel_error = 0
            ang_vel_error = 0
            

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class noise:
        add_noise = False
        noise_level = 0.2# scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.01
            ang_vel = 0.02
            gravity = 0.01
            height_measurements = 0.1
            
    class domain_rand:
        curriculum = []
        randomize_friction = True
        friction_range = [-0.1, 0.1]
        randomize_base_mass = True
        added_mass_range = [-2, 2]
        push_robots = True
        start_push_iter = 50
        push_interval_s = 4
        push_duration_s = 2
        push_interval_range = [1,1]
        push_duration_range = [0.5,1]
        min_push_vel_xy = [100.,100]
        max_push_vel_xy = [130.,130]
        impulse = False
        
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        multiple = False
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 0.5
        dynamic_friction = 0.5
        restitution = 0.
        # rough terrain only:
        rough_plane = True
        roughness = 0.025
        
        slope = [-0.3,0.3]
        slope_portion = 0. #this determines how many slopes to generate
        slope_roughness = 0.01
        
        rough_stair = False
        stair_size = [1.5,0.35]
        
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0., 1.0 , 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # measure_heights = False
class A1WXRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.5
        # init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'lrelu'
        
        RMA=True
        RMA_hidden_dims=[128,128]
        privilege_obs_dim = 12
        num_hist_obs = A1WXRoughCfg.env.num_hist_obs
        num_privilege_obs = privilege_obs_dim*num_hist_obs
        num_latent = 8
        
        # estimator=True
        estimator=False
        
        decouple = True
        dog = False
        dog_policy_path = "/home/ravenhuang/amp/locomani/legged_gym/logs/eval/Feb02_04-35-34_push_walk_dog_only/model_30150.pt"
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        adaptive_weight_alpha = 0.9991
        adaptive_weight_beta = 0.998
        start_alpha = 0.8
        end_alpha = 0.3
        start_beta = 0.5
        end_beta = 0.05
        
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24
        load_run = "Feb02_10-24-59_push_walk_dog_finetuned_arm"
        resume = True
        reset_std = True
        load_optimizer = False
        checkpoint = -1
        max_iterations = 15000 # number of policy updates
        experiment_name = 'a1wx'
        run_name = 'push_walk_dog_finetuned_arm_bc'
        

  