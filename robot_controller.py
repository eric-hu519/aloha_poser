from calendar import c
import time
from tkinter import OFF
from turtle import left
from unittest.mock import DEFAULT
from charset_normalizer import detect
import test
import tqdm
from flask import g
from httpx import head
from traitlets import default
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import sys
import json
import logging
import re
import jsonschema
from torch import NoneType, le
from requests.exceptions import Timeout, RequestException
import os
import zlib
import requests
import cv2
import threading

from realsense_test_aloha import RealSenseCamera

SCALE = 0.01
STRAW_POS = [243,381,628,33,145]
CUP_POS = [465,367,1093,81,144]

# STRAW_POS = [465,367,628,33,145]
# CUP_POS = [243,381,1093,81,144]

IMAGE_URL = "http://192.168.31.109:1115/upload"
CALIBRITION_FILE_PATH_L = '/home/mamager/interbotix_ws/src/aloha/act-plus-plus/aloha_poser/calibration/calib_result_20241122222424'
CALIBRITION_FILE_PATH_R = '/home/mamager/interbotix_ws/src/aloha/act-plus-plus/aloha_poser/calibration_right/calib_result_20241125001140'
OFFSET_L = [0.030, -0.15, 0]
OFFSET_R = [-0.28, 0.075, 0.25]
DEFAULT_SIDE = 'right'
GRIPPER_POSE_THRESHOLD = 60
PREP_GRIPPER_POSE = [0.8, 0.785] #roll and pitch

HOME_POSE = {'left': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0},
             'right': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0}}

class robot_controller:
    def __init__(self,one_arm_mode = True, test_camera = False) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        #init robot
        self.puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_left', init_node=True)
        self.puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_right', init_node=False)
        
        self.puppet_bot_left.dxl.robot_torque_enable("group", "arm", True)
        self.puppet_bot_left.dxl.robot_torque_enable("single", "gripper", True)
        
        self.puppet_bot_right.dxl.robot_torque_enable("group", "arm", True)
        self.puppet_bot_right.dxl.robot_torque_enable("single", "gripper", True)
        
        self.one_arm_mode = one_arm_mode
        #init robot status
        self.side = None
        self.target_pos = {}
        self.detect_result = dict()
        self.robot_status = dict()
        self.robot_status['left'] = {'gripper': 'open', 'holding_status': False}
        self.robot_status['right'] = {'gripper': 'open', 'holding_status': False}
        
        #init camera
        self.test_camera = test_camera
        self.predefined_pos = {'red_straw': STRAW_POS, 'plastic_cup': CUP_POS}
        if not self.test_camera:
            self.camera = RealSenseCamera()
            self.camera.start()
            self.camera_intrinsic = self.camera.get_camera_intrinsic()
            #self.logger.info(f"Camera intrinsic: {self.camera_intrinsic}")
            self.camera.stop()
        else:
            self.camera_intrinsic = (381.6689147949219, 381.6689147949219, 321.3359680175781, 236.79440307617188)

    def set_pose(self, args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.set_ee_pose_components(**args)
        
    def image_to_robot_coords(self,target_pos, camera_pose, depth_scale):
        # 将图像坐标 (x, y) 和深度值转换为相机坐标
        z_c = target_pos[2] * depth_scale
        x_c = (target_pos[0] - self.camera_intrinsic[2]) * z_c / self.camera_intrinsic[0]
        y_c = (target_pos[1] - self.camera_intrinsic[3]) * z_c / self.camera_intrinsic[1]
        camera_coords = np.array([x_c, y_c, z_c, 1.0])
        # 将相机坐标转换为机械臂坐标
        robot_coords = np.dot(camera_pose, camera_coords)
        return robot_coords[:3]
    
    def load_calib_file(self,file_path):
        camera_pose = np.loadtxt(f'{file_path}/realsense_camera_pose.txt')
        depth_scale = np.loadtxt(f'{file_path}/realsense_camera_depth_scale.txt')
        return camera_pose, depth_scale
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

        return roll, pitch
    def get_target_coord_by_side(self,side,target_pos):
        if side == 'left':
            camera_pose, depth_scale = self.load_calib_file(CALIBRITION_FILE_PATH_L)
            offsets = OFFSET_L
            #get roll and pitch from matrix
            roll, pitch = self.get_roll_pitch_yaw_from_matrix(self.puppet_bot_left.arm.get_ee_pose())
        else:
            camera_pose, depth_scale = self.load_calib_file(CALIBRITION_FILE_PATH_R)
            offsets = OFFSET_R
            #get roll and pitch from matrix
            roll, pitch = self.get_roll_pitch_yaw_from_matrix(self.puppet_bot_right.arm.get_ee_pose())
            
        #offset depth if holding(so that arm can reach the center of target)
        if target_pos[3] >= GRIPPER_POSE_THRESHOLD and self.robot_status[side]['holding_status'] == True:
            target_pos[2] = target_pos[2] + target_pos[3]/3
        #convert image to robot coords
        robot_coords = self.image_to_robot_coords(target_pos[:3], camera_pose, depth_scale)
        #add offsets
        for i in range(len(offsets)):
            robot_coords[i] = robot_coords[i] + offsets[i]
        #adjust gripper position according to bbox width and holding status
        if self.robot_status[side]['holding_status'] == False:
            if target_pos[3] >= GRIPPER_POSE_THRESHOLD:
                robot_coords = np.append(robot_coords, PREP_GRIPPER_POSE)
            else:
                robot_coords = np.append(robot_coords,[0,0])
        else:
            #get current gripper pose
            robot_coords = np.append(robot_coords, [roll, pitch])
        #self.logger.info(f"robot position: {robot_coords}")
        return robot_coords

    def detect(self, args, side=None):
        #push target obj to tracker
        if not self.test_camera:
            #use real camera if not test
            self.push_nlp_to_visiontracker(args['target'])
            #get_image_from dual sense camera
            depth_img, color_img = self.get_rgbd_img()
            response = self.send_rgb_img(color_img,IMAGE_URL)
        #convert utf-8 to json
            response = json.loads(response.content.decode('utf-8'))
        #get object position from tracker
            x_center, y_center, width, height = response['box']
        #print(f"object position: {x_center},{y_center},{width},{height}")
            
        #print(f"center point: {x_center},{y_center}")
        #get deepth values
            depth = depth_img[int(y_center),int(x_center)]
            #self.logger.info(f"{args['target']} position: {x_center},{y_center},{depth},{width},{height}")
        else:
            #use predifined position if test
            
            x_center, y_center, depth, width, height = self.predefined_pos[args['target']]
        #calculate target position
        target_pos = [x_center, y_center, depth, width, height]
        #update target position and assign sides automatically
        if self.target_pos == {}:
            #if no target detected before, assign target according to current target
            self.target_pos[args['target']] = target_pos
            suggested_side = 'left' if x_center < 320 else 'right'
            left_detect_result = self.get_target_coord_by_side('left', target_pos)
            right_detect_result = self.get_target_coord_by_side('right', target_pos)
            default_side = left_detect_result if suggested_side == 'left' else right_detect_result
        else:
            #TODO support multiple targets
            assert len(self.target_pos) == 1, "Multiple targets are not supported yet."
            #if there are other targets, compare the new target's x_center with the previous one
            for target in self.target_pos:
                #get the previous target's x_center
                pre_x = self.target_pos[target][0]
                if x_center < pre_x:
                    suggested_side = 'left'
                    #check if the left side is occupied
                    if self.robot_status['left']['holding_status'] == True and self.robot_status['right']['holding_status'] == False:
                        suggested_side = 'right'
                else:
                    suggested_side = 'right'
                    #check if the right side is occupied
                    if self.robot_status['right']['holding_status'] == True and self.robot_status['left']['holding_status'] == False:
                        suggested_side = 'left'
            #update detect result
            left_detect_result = self.get_target_coord_by_side('left', target_pos)
            right_detect_result = self.get_target_coord_by_side('right', target_pos)
            default_side = left_detect_result if suggested_side == 'left' else right_detect_result
        #update detect result
        self.detect_result[args['target']] = {'left': left_detect_result, 'right': right_detect_result, 'default_side': default_side, 'suggested_side': suggested_side}
        self.logger.info(f"Detect result: {self.detect_result[args['target']]}")

    def push_nlp_to_visiontracker(self, nlp):#将nlp消息发送给物理机2
        url = 'http://192.168.31.109:1115/vision'  # 物理机2的地址
        headers = {'Content-Type': 'application/json'}  # 设置HTTP头部为JSON
        data = {'nlp': nlp}  # 将nlp消息包装成一个字典
        try:
            response = requests.post(url, headers=headers, json=data)  # 发送POST请求
            response.raise_for_status()
            self.logger.info('Successfully pushed the message to the other machine.')
        except (ConnectionError, RequestException) as e:
            raise(f'Failed to push the message to the other machine with error: {e}')


    def get_rgbd_img(self):
        self.camera.start()
        depth_image, color_image = self.camera.get_frames()
        self.camera.stop()
        return depth_image, color_image

    def send_rgb_img(self,img,url):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        compressed_img = zlib.compress(img)
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            response = requests.post(url, data=compressed_img, headers=headers, timeout = 10)
            response.raise_for_status()
        except (Timeout, RequestException) as e:
            raise e
        return response
        


    def set_joint_pose(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        joint_name = args['joint_name']
        value = args['value']
        if args.get('value_type') == 'radius':
            value = np.pi * value
        bot.arm.set_single_joint_position(joint_name, value)
        

    def open_gripper(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #update robot status
        self.robot_status[side]['gripper'] = 'open'
        self.robot_status[side]['holding_status'] = False
        #execute open gripper
        bot.gripper.open()

    def close_gripper(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #update robot status
        self.robot_status[side]['gripper'] = 'close'
        self.robot_status[side]['holding_status'] = True
        #execute close gripper
        bot.gripper.close(side = side)

    def set_trajectory(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.set_ee_cartesian_trajectory(**args)

    def home_pose(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.set_ee_pose_components(**HOME_POSE[side])

    def sleep_pose(self,args, side):
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.go_to_sleep_pose()

    def get_function_by_name(self,actuator_type, name):
        if actuator_type == 'camera':
            return {
                'detect': self.detect
            }.get(name, None)
        if actuator_type == 'arm':
            return {
                'set_pose': self.set_pose,
                'set_joint_pose': self.set_joint_pose,
                'set_trajectory': self.set_trajectory,
                'home_pose': self.home_pose,
                'sleep_pose': self.sleep_pose,
                
            }.get(name, None)
        elif actuator_type == 'gripper':
            return {
                'open': self.open_gripper,
                'close': self.close_gripper
            }.get(name, None)
    def parse_args(self, args, side = None):
        parsed_args = {}
        pattern = r'(detect_result)\[["\']?(\w+)["\']?\]\[(\d+)\](\s*[\+\-]?\s*\d+(\.\d+)?(cm)?)?'
        side_pattern = r'detect_result\[["\']?(\w+)["\']?\]\[["\']?(\w+)["\']?\]'
        if side is not None and not self.one_arm_mode:
            side_match = re.match(side_pattern,side)
            side = self.detect_result[side_match.group(1)][side_match.group(2)]
            print(f"side: {side}")
        if 'target' in args:
            if args['target'] is not None:
                parsed_args['target'] = args['target']
        for arg in args:
            if arg == 'target':
                break
            if args[arg] is not None and isinstance(args[arg],str):
                if "detect_result" in args[arg]:
                    args[arg] = args[arg].replace(" ","")
                    match = re.match(pattern,args[arg])
                    if match.group(4) is not None:
                        scale_pattern = r'([\+\-\+]?\d+(\.\d+)?)'
                        
                        
                        scale_match = re.match(scale_pattern,match.group(4).strip())
                        diviation = scale_match.group(1)
                    else:
                        diviation = 0
                    #get detect result according to side
                    if self.one_arm_mode:
                        side = DEFAULT_SIDE 
                    if side is None:
                        parsed_args[arg] = self.detect_result[match.group(2)]['default_side'][int(match.group(3))] + (SCALE*float(diviation))
                        side = self.detect_result[match.group(2)]['suggested_side']
                    else:
                        parsed_args[arg] = self.detect_result[match.group(2)][side][int(match.group(3))] + (SCALE*float(diviation))
                    #parsed_args[arg] = parsed_args[arg][match.group(2)]
                    #parsed_args[arg] = eval(match.group(1),{"detect_result":self.detect_result})[match.group(2)][int(match.group(3))] + (SCALE*float(scale_match.group(1)) if match.group(4) is not None else 0)
            else:
                parsed_args[arg] = args[arg]
        return parsed_args, side
    def run(self,action_sequence):
        #print current working directory
        #self.logger.info(f"Current working directory: {os.getcwd()}")
        if '.json' in action_sequence:
            with open(action_sequence, 'r') as file:
                data = json.load(file)
        else:
            data = action_sequence
        self.logger.info(f"Action length: {len(data)}")
        #init to home pose
        if self.one_arm_mode:
            self.home_pose({}, DEFAULT_SIDE)
            self.open_gripper({}, DEFAULT_SIDE)
        else:
            self.init_thread_left = threading.Thread(target=self.init_pose_for_one_side, args=('left',))
            self.init_thread_right = threading.Thread(target=self.init_pose_for_one_side, args=('right',))
            self.init_thread_left.start()
            self.init_thread_right.start()
            self.init_thread_left.join()
            self.init_thread_right.join()

        if not self.one_arm_mode:
        #separate actions by side and detect
            side1_actions, side2_actions, detect_actions = self.separate_actions(data)
            #run action sequence by detect-both sides-detect-both sides... 
            #so that the robot can operate synchronously
            for item in detect_actions:
                #run detect action first
                self.run_one_side(item)
                side1_sub_actions = side1_actions.pop(0)
                side2_sub_actions = side2_actions.pop(0)
                self.side1_action_thread = threading.Thread(target=self.run_one_side, args=(side1_sub_actions,))
                self.side2_action_thread = threading.Thread(target=self.run_one_side, args=(side2_sub_actions,))
                self.side1_action_thread.start()
                self.side2_action_thread.start()
                #wait for both sides to finish
                self.side1_action_thread.join()
                self.side2_action_thread.join()
            #set both arms to sleep pose
            self.sleep_thread_left = threading.Thread(target=self.sleep_pose, args=({}, 'left'))
            self.sleep_thread_right = threading.Thread(target=self.sleep_pose, args=({}, 'right'))
            self.sleep_thread_left.start()
            self.sleep_thread_right.start()
            self.sleep_thread_left.join()
            self.sleep_thread_right.join()

        else:
            #run action sequence for one side
            self.run_one_side(data)
            self.sleep_pose({}, DEFAULT_SIDE)
        
    def init_pose_for_one_side(self,side):
        self.home_pose({}, side)
        self.open_gripper({}, side)

    def run_one_side(self,action_sequence):
        #run action sequence
        for item in action_sequence:
            actuator_type = item.get('type')
            name = item.get('name')
            args = item.get('args')
            side = item.get('side') if item.get('side') is not None else None
            if args is not None:
                args, run_side = self.parse_args(args,side)
            #assign side to operate when detect is not called
            func = self.get_function_by_name(actuator_type, name)
            self.logger.info(f"Running {actuator_type} {name} with args {args} for {run_side} arm")
            func(args, run_side)
        

    #separete actions by side and detect. 
    #ensure the synchronization of actions
    def separate_actions(self,action_sequence):
        #separate actions by side
        side1_actions = []
        side2_actions = []
        detect_actions = []
        i=0
        while i < len(action_sequence):
            if action_sequence[i]['type'] == 'camera' and action_sequence[i]['name'] == 'detect':
                detect_actions.append([action_sequence[i]])
                side1_actions.append([])
                side2_actions.append([])
                #check the following sequence, if the next action is camera, skip it
                for j in range(i+1,len(action_sequence)):
                    if action_sequence[j]['type'] == 'camera':
                        detect_actions[-1].append(action_sequence[j])
                    else:
                        break
                #skip the following camera actions
                i = j
                continue
            else:
                #if side1 is empty, add action to side1
                if len(side1_actions) == 0:
                    side1_actions.append([action_sequence[i]])
                elif len(side1_actions[-1]) == 0:
                    side1_actions[-1].append(action_sequence[i])
                else:
                    #save to side1 if the side are the same
                    if action_sequence[i]['side'] == side1_actions[-1][0]['side']:
                        side1_actions[-1].append(action_sequence[i])
                    else:
                        if side2_actions == []:
                            side2_actions.append([action_sequence[i]])
                        else:
                            side2_actions[-1].append(action_sequence[i])
            i = i + 1
        return side1_actions, side2_actions, detect_actions
        
def main():
    controller = robot_controller(one_arm_mode = False, test_camera = True)
    controller.run('two-arm-json/grasp_straw_human.json')
    #controller.run('detect_two_arm.json')
    # action_sequence = '/home/mamager/interbotix_ws/src/aloha/act-plus-plus/aloha_poser/two-arm-json/grasp_straw_human.json'
    # with open(action_sequence, 'r') as file:
    #     data = json.load(file)
    # print(f'Action length: {len(data)}')
    # controller.separate_actions(data)
    # controller.run('action_sequence_test.json')
    #controller.detect('red_straw')
    #print(controller.detect_result['red_straw'][0])

if __name__ == "__main__":
    main()