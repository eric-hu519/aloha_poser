from pickle import LIST
import re
import numpy as np
from traitlets import Int
from scipy.spatial.transform import Rotation as R

def depth_circle_sampler(radius,depth_img, x_center,y_center):
    """
    sample a circle around x_center and y_center and return the average depth value in the circle
    """
    x = np.arange(depth_img.shape[1])
    y = np.arange(depth_img.shape[0])
    xx, yy = np.meshgrid(x, y)
    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2 < radius ** 2
    circle_depth = depth_img[circle]
    return np.mean(circle_depth)

def check_grasp_pos(bot, grasp_pos, check_range=[-0.02, 0.08, 0.01]):
    """
    Check joint limit and pos reachability

    :param bot: interbotix robot arm
    :param grasp_pos: grasp pos list
    :param check_range: check range
    """
    #grasp_pos['execute'] = False
    original_grasp_pos = grasp_pos.copy()
    for i in np.arange(check_range[0], check_range[1], check_range[2]):
        grasp_pos[2] = original_grasp_pos[2] + i
        _,is_valid = bot.arm.check_ee_pose_components(grasp_pos[0], grasp_pos[1], grasp_pos[2], grasp_pos[3], grasp_pos[4], grasp_pos[5])
        if not is_valid:
            return is_valid
    return is_valid
    

def post_process_grasp_pose(grasp_pose, offset):
    """
    Post process grasp pose by adding offset to the position and adjusting orientation
    
    :param grasp_pose: grasp pose to be processed
    :param offset: offset to be added to the position
    """
    rot = grasp_pose.rotation_matrix
    trans = grasp_pose.translation
    #gripper = gripper.transform(trans_mat)
    #trans = gripper.get_center()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans

    r = R.from_matrix(rot)
    gripper_rot_xyz = r.as_euler('xyz', degrees=True)
    #process rotation
    if gripper_rot_xyz[0] > 90:
        gripper_rot_xyz[0] -= 180
    elif gripper_rot_xyz[0] < -90:
        gripper_rot_xyz[0] += 180
    
    if gripper_rot_xyz[1] > 90:
        gripper_rot_xyz[1] -= 180
    elif gripper_rot_xyz[1] < -90:
        gripper_rot_xyz[1] += 180

    if gripper_rot_xyz[2] > 90:
        gripper_rot_xyz[2] -= 180
    elif gripper_rot_xyz[2] < -90:
        gripper_rot_xyz[2] += 180
    args = {}
    args['x'] = trans[0] + offset[0]
    args['y'] = trans[1] + offset[1]
    args['z'] = trans[2] + offset[2]
    args['roll'] = np.deg2rad(gripper_rot_xyz[0]) + offset[3]
    args['pitch'] = np.deg2rad(gripper_rot_xyz[1]) + offset[4]
    args['yaw'] = np.deg2rad(gripper_rot_xyz[2]) + offset[5]

    args_list = [args['x'], args['y'], args['z'], args['roll'], args['pitch'], args['yaw']]
    return args_list

def args_to_ee_matrix(args):
    """
    Convert args to end effector pose matrix
    :param args: args list
    """
    x, y, z, roll, pitch, yaw = args
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    T[:3, :3] = r.as_matrix()
    return T
def ee_matrix_to_args(T):
    """
    Convert end effector pose matrix to args list
    :param T: end effector pose matrix
    """
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    x, y, z = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    roll, pitch, yaw = get_roll_pitch_yaw_from_matrix(r.as_matrix())
    return [x, y, z, roll, pitch, yaw]
def apply_local_offset_to_pose(T_ee: np.ndarray, delta_local: np.ndarray) -> np.ndarray:
    """
    给定末端执行器在全局坐标系下的 4×4 齐次变换矩阵 T_ee 和在局部坐标系下的偏移向量 delta_local，
    返回应用偏移后的新的 4×4 齐次变换矩阵（仍在全局坐标系下）
    
    Parameters:
    - T_ee: np.ndarray, shape=(4, 4)，当前末端位姿
    - delta_local: np.ndarray, shape=(3,)，局部坐标系下的偏移向量（如 [0, 0.1, 0] 表示沿 y+ 偏移 10cm）
    
    Returns:
    - T_ee_new: np.ndarray, shape=(4, 4)，偏移后的新末端位姿（全局坐标系下）
    """
    #assert T_ee.shape == (4, 4)
    #assert delta_local.shape == (3,)
    
    # 构造局部偏移的 4x4 齐次变换矩阵（平移 + 单位旋转）
    T_offset_local = np.eye(4)
    T_offset_local[:3, 3] = delta_local

    # 右乘，实现“在局部坐标系下”的偏移
    T_ee_new = T_ee @ T_offset_local

    return T_ee_new

def get_roll_pitch_yaw_from_matrix(matrix):
    """
    从变换矩阵中提取 roll, pitch 和 yaw 角度
    """
    # 提取旋转矩阵 R
    R = matrix[:3, :3]

    # 计算 pitch
    pitch = np.arcsin(-R[2, 0])

    # 计算 roll 和 yaw
    if np.cos(pitch) != 0:
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0

    return roll, pitch, yaw
from collections import deque
from dataclasses import dataclass
from typing import List
@dataclass
class RobotAction:
    #stepid是int类型，表示动作的序号
    stepID: int
    is_grasping: bool
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class RobotStatusLogger:
    def __init__(self, max_length=10000,robot = None):
        self.history = deque(maxlen=max_length)  # 自动清除最旧记录防止内存过大
        self.robot = robot
        self.is_grasping = False
    def log_action(self, stepid):
        """
        记录机械臂动作
        """
        x, y, z, roll, pitch, yaw = self.get_arm_position()
        action = RobotAction(
            stepID = stepid,
            is_grasping=self.is_grasping,
            x = x,
            y = y,
            z = z,
            roll = roll,
            pitch = pitch,
            yaw = yaw
        )
        self.history.append(action)

    def get_arm_position(self):
        """
        获取机械臂末端位置
        """
        #get transform matrix
        pose_matrix =  self.robot.arm.get_ee_pose()
        ee_pose = pose_matrix[:3,3]
        x = ee_pose[0]
        y = ee_pose[1]
        z = ee_pose[2]
        roll, pitch, yaw = self.get_roll_pitch_yaw_from_matrix(pose_matrix)
        return x, y, z, roll, pitch, yaw
        
    def get_roll_pitch_yaw_from_matrix(self, matrix):
        """
        从变换矩阵中提取 roll, pitch 和 yaw 角度
        """
        # 提取旋转矩阵 R
        R = matrix[:3, :3]

        # 计算 pitch
        pitch = np.arcsin(-R[2, 0])

        # 计算 roll 和 yaw
        if np.cos(pitch) != 0:
            roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
            yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0

        return roll, pitch, yaw
    def get_last_action(self):
        return self.history[-1] if self.history else None

    def get_action_by_index(self, index):
        if -len(self.history) <= index < len(self.history):
            return self.history[index]
        return None

    # def query_by_time_range(self, start_time, end_time):
    #     return [action for action in self.history
    #             if start_time <= action.timestamp <= end_time