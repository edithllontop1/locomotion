from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import torch
import modern_robotics as mr
import time
# This script makes the end-effector perform pick, pour, and place tasks
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'
class obs_scales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


def torch_rand_float(lower, upper, shape):
    return (upper - lower) * torch.rand(*shape) + lower

class wx200():
    def __init__(self,policy_path,hist_len=5,control_frequency=100):
        self.hist_len = hist_len
        self.default_dof_pos = torch.tensor([0,-1.88,1.5,0.8,0,0,0])
        self.robot = InterbotixManipulatorXS("wx200", "arm", "gripper")
        self.failed = 0
        self.iter  = 0
        self.last_vel = [torch.zeros((1,7),dtype=torch.float) for _ in range(7)]
        self.policy = torch.jit.load(policy_path,map_location='cpu')
        self.actions = torch.zeros((1,7))
        self.dt = 1/control_frequency

    def capture_joint_velocities(self):
        joint_commands = []
        for name in self.robot.arm.group_info.joint_names:
            joint_commands.append(self.robot.arm.core.joint_states.velocity[self.robot.arm.core.js_index_map[name]])
        joint_commands += [0,0]
        return torch.tensor(joint_commands).unsqueeze(0)

    def capture_joint_positions(self):
        joint_commands = []
        for name in self.robot.arm.group_info.joint_names:
            joint_commands.append(self.robot.arm.core.joint_states.position[self.robot.arm.core.js_index_map[name]])
        joint_commands += [0,0]
        return torch.tensor(joint_commands).unsqueeze(0)
                
    def get_observation(self,commands,actions):
        dof_pos = self.capture_joint_positions()
        try:
            dof_vel = self.capture_joint_velocities()
            self.last_vel = dof_vel
        except:
            print("use the last vel")
            self.failed += 1
            dof_vel = self.last_vel
            
        obs_buf = torch.cat((  
                    commands,
                    (dof_pos - self.default_dof_pos) * obs_scales.dof_pos,
                    dof_vel * obs_scales.dof_vel,
                    actions
                    ),dim=-1)
        
        # hist_obs.pop(0)
        # hist_obs.append(obs_buf)
        # obs_buf = torch.cat(hist_obs[:hist_len],dim=-1)
        return obs_buf
    
    def reach(self,commands, time_out = 5):
        self.robot.arm.go_to_sleep_pose()
        begin = time.time()
        while (time.time()-begin) < time_out:
            self.iter += 1
            start = time.time()
            obs_buf = self.get_observation(commands,self.actions)
            self.actions = self.policy(obs_buf)
            duration = time.time()-start
            if duration < self.dt:
                time.sleep(self.dt-duration)
            self.robot.arm.set_joint_positions(self.actions.detach().numpy()[0,:5], blocking=False)
            # time.sleep(0.001)
            # print("duration",time.time()-start)
        
        print("eepos",commands,self.robot.arm.get_ee_pose()[:3,3])    
        self.robot.arm.go_to_sleep_pose()



def main():
   
    path = "/home/ravenhuang/locomani/locomani/policywx.pt"
    wx200_robot = wx200(path)
   
    for _ in range(2):
        commands = torch_rand_float(torch.tensor([-0.18,-0.18,0.2]),torch.tensor([0.18,0.18,0.25]),(1,3))
        wx200_robot.reach(commands)
        
    
    print(wx200_robot.failed/wx200_robot.iter)
    

if __name__=='__main__':
    main()