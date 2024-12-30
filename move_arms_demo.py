from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import sys
from aloha_scripts.robot_utils import move_arms, torque_on, torque_off, move_grippers
from aloha_scripts.real_env import make_real_env

PUPPET_GRIPPER_JOINT_OPEN =0.0
PUPPET_GRIPPER_JOINT_CLOSE =-1.5

def left_puppet_demo():

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_left', init_node=True)

    torque_on(puppet_bot_left) #开启夹爪前必须激活夹爪！！！

    puppet_bot_left.arm.set_ee_pose_components(x=0.2, z=0.4)    # puppet_bot_left.arm.set_ee_pose_components(x=0,y=0.2, z=0.2, roll=0.0, pitch=0, yaw=0.0)
    # puppet_bot_left.arm.set_ee_pose_components(x=0.3, z=0.2)
    puppet_bot_left.gripper.close()
    puppet_bot_left.arm.set_ee_pose_components(x=0.2, z=0.4)
    puppet_bot_left.arm.set_single_joint_position("waist", -np.pi/2.0) #base turn left 90
    #puppet_bot_left.gripper.open()
    puppet_bot_left.arm.set_ee_cartesian_trajectory(x=0.3, z=0.2, roll=0.0, pitch=0, yaw=0.0)
    
    puppet_bot_left.gripper.close()
    puppet_bot_left.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    puppet_bot_left.arm.set_single_joint_position("waist", -np.pi/2.0) #base turn right 180
    puppet_bot_left.arm.set_ee_cartesian_trajectory(pitch=1.5) #moduan turn down 
    puppet_bot_left.arm.set_ee_cartesian_trajectory(pitch=-1.5) #moduan turn up
    puppet_bot_left.arm.set_single_joint_position("waist", np.pi/2.0)
    puppet_bot_left.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    puppet_bot_left.gripper.open()
   
    puppet_bot_left.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    puppet_bot_left.arm.go_to_home_pose()
    puppet_bot_left.arm.go_to_sleep_pose()

def left_right_puppet_demo():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_right', init_node=True)
                                                    

if __name__ =="__main__":
    left_puppet_demo()