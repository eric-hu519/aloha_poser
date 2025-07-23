#####
#robot controller v2 integrated with anygrasp sdk 
#and deleted any redundant code in v1
#####
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import json
import logging
import re
import argparse
from requests.exceptions import Timeout, RequestException
import zlib
import requests
import cv2
import threading
from utils.robot_utils import depth_circle_sampler
from scipy.spatial.transform import Rotation as R
from gsnet import AnyGrasp
#from realsense_test_aloha import RealSenseCamera
from utils.pressure_sensor import PressureSensor
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import RealSenseCapture
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import CloudPointProcessor as processor
IMAGE_URL = "http://192.168.31.109:1115/upload"
OFFSET_L = [0.07, -0.075, 0.25]
OFFSET_R = [0.07, 0.075, 0.25]#NOTE: Need recalibrite
DEFAULT_SIDE = 'left'
HOME_POSE = {'left': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0},
             'right': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0}}
GRIPPER_POSE_THRESHOLD = 60
PRESSURE_SENSOR_THRESHOLD = 0.5

SCALE = 0.01
CAM_ROT = [-56,0,180]
CAM_TRANS = [0,0.45,0]
BASE_ROT = [126,0,90]
BASE_TRANS = [0.32,0.31,-0.24]

class Robot_Controller:
    def __init__(self,one_arm_mode = True, test_camera = False, enable_pressure = False) -> None:
        """
        :param one_arm_mode: bool, if True, only one arm will be used
        :param test_camera: bool, if True, the realsense camera will not be used, use presaved img instead
        :param enable_pressure: bool, if True, the pressure sensor will be used
        """
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
        self.detect_target = dict()
        self.robot_status['left'] = {'gripper': 'open', 'holding_status': False}
        self.robot_status['right'] = {'gripper': 'open', 'holding_status': False}
        self.robot_status['left']['gripper_pressure'] = None
        self.robot_status['right']['gripper_pressure'] = None
        self.robot_status['left']['robot_pose'] = []
        self.robot_status['right']['robot_pose'] = []
        self.detect_target['left'] = []
        self.detect_target['right'] = []
        #action target is different from detect target, it is parsed from action sequence
        self.action_target =  []
        self.holding_obj = None
        self.enable_pressure = enable_pressure
        #init camera
        self.test_camera = test_camera
        #init anygrasp sdk
        self.grasp_processor = Anygrasp_Processor(side=DEFAULT_SIDE)
        
    def run(self,action_sequence):
        """
        运行动作序列
        """
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
            #self.sleep_pose({}, DEFAULT_SIDE)
        
    def run_one_side(self,action_sequence):
        """
        运行单边动作序列
        """
        #run action sequence
        for item in action_sequence:
            actuator_type = item.get('type')
            name = item.get('name')
            args = item.get('args')
            side = item.get('side') if item.get('side') is not None else None
            if args is not None:
                args, run_side, detect_target = self.parse_args(args,side)
            #assign side to operate when detect is not called
            func = self.get_function_by_name(actuator_type, name)
            if detect_target is not None:
                self.action_target.append(detect_target)
            self.logger.info(f"Running {actuator_type} {name} with args {args} for {run_side} arm")
            func(args, run_side)
            #update target position by arm position if grasp is called
            if actuator_type != 'camera':
                assert detect_target is not None, "Detect target is not assigned."
                if name == 'grasp':
                    self.detect_result[detect_target][run_side] = self.robot_status[run_side]['robot_pose'][-1]
                    self.holding_obj = detect_target
                #if holding, target coordinate changes with robot arm
                elif name == 'set_pose' and self.robot_status[run_side]['holding_status'] == True and self.holding_obj == detect_target:
                    self.detect_result[detect_target][run_side] = self.robot_status[run_side]['robot_pose'][-1]
                elif name == 'vertical_pose' and self.robot_status[run_side]['holding_status'] == True and self.holding_obj == detect_target:
                    self.detect_result[detect_target][run_side] = self.robot_status[run_side]['robot_pose'][-1]
                elif name == 'horizontal_pose' and self.robot_status[run_side]['holding_status'] == True and self.holding_obj == detect_target:
                    self.detect_result[detect_target][run_side] = self.robot_status[run_side]['robot_pose'][-1]
                elif name == 'release':
                    self.detect_result[self.holding_obj][run_side] = self.robot_status[run_side]['robot_pose'][-2]
                    self.holding_obj = None
                    
    def init_pose_for_one_side(self,side):
        """
        单边机械臂初始化
        """
        self.home_pose({}, side)
        self.open_gripper({}, side)
    def parse_args(self, args, side = None):
        """
        解析参数，将参数中的detect_result替换为检测结果
        """
        parsed_args = {}
        detect_target = None
        pattern = r'(detect_result)\[["\']?(\w+)["\']?\]\[(\d+)\](\s*[\+\-]?\s*\d+(\.\d+)?(cm)?)?'
        side_pattern = r'detect_result\[["\']?(\w+)["\']?\]\[["\']?(\w+)["\']?\]'
        if side is not None and not self.one_arm_mode:
            side_match = re.match(side_pattern,side)
            side = self.detect_result[side_match.group(1)][side_match.group(2)]
        else:
            side = DEFAULT_SIDE
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
                    detect_target = match.group(2)
            else:
                parsed_args[arg] = args[arg]
        return parsed_args, side, detect_target

    def get_function_by_name(self,actuator_type, name):
        """
        根据名称获取对应的API函数
        """
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
                'grasp': self.grasp,
                'release': self.release_obj,
                'vertical_pose': self.restore_vertical,
                'horizontal_pose': self.restore_horizontal
            }.get(name, None)
        elif actuator_type == 'gripper':
            return {
                'open': self.open_gripper,
                'close': self.close_gripper
            }.get(name, None)
        
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
    def push_nlp_to_visiontracker(self, nlp):#将nlp消息发送给物理机2
        """
        将检测目标信息发给视觉追踪模块
        """
        url = 'http://192.168.31.109:1115/vision'  # 物理机2的地址
        headers = {'Content-Type': 'application/json'}  # 设置HTTP头部为JSON
        data = {'nlp': nlp}  # 将nlp消息包装成一个字典
        try:
            response = requests.post(url, headers=headers, json=data)  # 发送POST请求
            response.raise_for_status()
            self.logger.info('Successfully pushed the message to the other machine.')
        except (ConnectionError, RequestException) as e:
            raise(f'Failed to push the message to the other machine with error: {e}')

    def send_rgb_img(self,img,url):
        """
        发送RGB图像到视觉追踪模块
        """
        img = cv2.imencode('.jpg', img)[1].tobytes()
        compressed_img = zlib.compress(img)
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            response = requests.post(url, data=compressed_img, headers=headers, timeout = 10)
            response.raise_for_status()
        except (Timeout, RequestException) as e:
            raise e
        return response

    def update_arm_position(self,side):
        """
        获取机械臂末端位置
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #get transform matrix
        pose_matrix =  bot.arm.get_ee_pose()
        ee_pose = pose_matrix[:3,3]
        roll, pitch = self.get_roll_pitch_yaw_from_matrix(pose_matrix)
        #add roll and pitch to the end of the list
        ee_pose = np.append(ee_pose, [roll, pitch])
        self.robot_status[side]['robot_pose'].append(ee_pose)
        
    def check_gripper_status(self, side):
        """
        检查夹爪状态是否夹持成功
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #get pressure sensor status
        sensor = PressureSensor()
        result = sensor.run_and_get_result()
        if 'no_touch' in result:
            return False
        else:
            return True

    def get_function_by_name(self,actuator_type, name):
        """
        根据名称获取对应的API函数
        """
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
                'grasp': self.grasp,
                'release': self.release_obj,
                'vertical_pose': self.restore_vertical,
                'horizontal_pose': self.restore_horizontal
            }.get(name, None)
        elif actuator_type == 'gripper':
            return {
                'open': self.open_gripper,
                'close': self.close_gripper
            }.get(name, None)
        
    """
    Belows are the APIs to control the robot
    """
    def set_pose(self, args, side):
        """
        移动机械臂到达指定坐标API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.set_ee_pose_components(**args)
        #update robot status
        self.update_arm_position(side)
        
    def restore_vertical(self, args, side):
        """
        旋转夹爪至竖直位置
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        args['roll'] = 0
        args['pitch'] = 0
        self.set_pose(args, side)
        #update robot status
        self.update_arm_position(side)
    def restore_horizontal(self, args, side):
        """
        旋转夹爪至水平位置
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        
        args['roll'] = 0.9
        args['pitch'] = 1.2
        self.set_pose(args, side)
        #update robot status
        self.update_arm_position(side)

    def grasp(self, args, side):
        """
        夹取目标API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #first move above the target
        args['z'] = args['z'] + 0.15
        self.set_pose(args, side)
        #then move to the target
        args['z'] = args['z'] - 0.15
        self.set_pose(args, side)
        #close gripper
        self.close_gripper(args, side, type='grasp')
        #move up
        args['z'] = args['z'] + 0.2
        self.set_pose(args, side)
        #update robot status
        self.robot_status[side]['holding_status'] = True
    
    def release_obj(self, args, side):
        """
        释放目标API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        args['z'] = args['z'] + 0.05
        self.set_pose(args, side)
        #open gripper
        self.open_gripper(args, side)
        #move up
        args['z'] = args['z'] + 0.05
        self.set_pose(args, side)
        #update robot status
        self.robot_status[side]['holding_status'] = False

    def detect(self, args, side=None):
        """
        根据指定参数进行目标检测，分配对应机械臂
        """
        rs_pipline = RealSenseCapture(is_mask=True, use_anchor=True)
        #get image from camera
        if self.test_camera:
            #TODO:read color and depth img and mask img from file
            raise NotImplementedError
        else:
            color_img, depth_img, mask = rs_pipline.get_image()
        #mask img by target region
        responese = self.send_rgb_img(color_img,IMAGE_URL)
        #get detect result
        response = json.loads(response.content.decode('utf-8'))
        #get object position from tracker
        x_center, y_center, width, height = response['box']
        #convert to 4 point anchor and set
        rs_pipline.anchor = np.array([[x_center - width/2, y_center - height/2],[ x_center + width/2, y_center - height/2],[x_center + width/2, y_center + height/2],[x_center - width/2, y_center + height/2]])
        #get object position in 3D
        pcd = rs_pipline.get_pcd()
        #get transform matrix and processed pcd
        _, cam2base_mat = processor.process_pcd(pcd,
                                        rotate_deg=BASE_ROT,
                                        trans_pos=BASE_TRANS
                )
        trans_pcd, pcd_mat = processor.process_pcd(pcd,
                                            rotate_deg=CAM_ROT,
                                            trans_pos=CAM_TRANS,
                                            )
        #get grasp pose
        grasp_pose = self.grasp_processor.get_grasp_pose(trans_pcd, pcd_mat, cam2base_mat)
        #update detect result
        #one arm
        self.detect_result[args['target']] = {'default_side': grasp_pose, 'suggested_side': DEFAULT_SIDE, DEFAULT_SIDE: grasp_pose}
        self.logger.info(f"Detect result: {self.detect_result[args['target']]}")
        #update detect target
        self.detect_target[DEFAULT_SIDE].append(args['target'])

    def open_gripper(self,args, side):
        """
        张开夹爪API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #update robot status
        self.robot_status[side]['gripper'] = 'open'
        self.robot_status[side]['holding_status'] = False
        #execute open gripper
        bot.gripper.open()

    
    def close_gripper(self,args, side, type = 'normal'):
        """
        闭合夹爪API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        #execute close gripper
        
        #retry mechanism
        if type == 'grasp':
            if self.enable_pressure:
                retry = 0
                #record gripper pressure
                #init gripper status
                while self.check_gripper_status(side):
                    continue
                self.robot_status[side]['gripper_pressure'] = False
                while retry < 3:
                    bot.gripper.close(side = side)
                    if self.check_gripper_status(side):
                        self.robot_status[side]['gripper_pressure'] = True
                        self.robot_status[side]['gripper'] = 'close'
                        self.robot_status[side]['holding_status'] = True
                        break
                    else:
                        retry = retry + 1
                        #back to the last pose and re-detect and close gripper
                        if len(self.robot_status[side]['robot_pose']) >= 2:
                            last_pose = self.robot_status[side]['robot_pose'][-2]
                        else:
                            last_pose = self.robot_status[side]['robot_pose'][-1]
                        retry_args = {'x': last_pose[0], 'y': last_pose[1], 'z': last_pose[2], 'roll': last_pose[3], 'pitch': last_pose[4]}
                        self.set_pose(retry_args, side)
                        #open gripper
                        self.open_gripper({}, side)
                        #update detect result
                        last_target = self.detect_target[side][-1]
                        self.detect({'target': last_target}, side)
                        #retry close gripper
                        last_pose = self.robot_status[side]['robot_pose'][-1]
                        retry_args = {'x': last_pose[0], 'y': last_pose[1], 'z': last_pose[2], 'roll': last_pose[3], 'pitch': last_pose[4]}
                        self.set_pose(retry_args, side)
                        bot.gripper.close(side = side)
                        self.logger.warning(f"Retry {retry} times")
                if retry == 3:
                    self.logger.error(f"Failed to grasp {last_target} after 3 retries.")
                    self.open_gripper({}, side)
                    self.home_pose({}, side)
                    self.sleep_pose({}, side)
                    exit(1)
            else:
                bot.gripper.close(side = side)
                #update robot status
                self.robot_status[side]['gripper_pressure'] = False
                self.robot_status[side]['gripper'] = 'close'
                self.robot_status[side]['holding_status'] = True
        else: #else for 'normal'
            bot.gripper.close(side = side)
            #update robot status
            self.robot_status[side]['gripper_pressure'] = False
            self.robot_status[side]['gripper'] = 'close'
            self.robot_status[side]['holding_status'] = False #False is set because not holding

    def home_pose(self, args, side):
        """
        将机械臂移动到初始位置API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.set_ee_pose_components(**HOME_POSE[side])
        #update robot status
        self.update_arm_position(side)

    def sleep_pose(self,args, side):
        """
        将机械臂移动到休眠位置API
        """
        bot = self.puppet_bot_left if side == 'left' else self.puppet_bot_right
        bot.arm.go_to_sleep_pose()

class Anygrasp_Processor:
    def __init__(self,side = 'left') -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path',default='/home/mamager/interbotix_ws/src/aloha/aloha_poser/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='Model checkpoint path')
        parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
        parser.add_argument('--gripper_height', type=float, default=0.19, help='Gripper height')
        parser.add_argument('--top_down_grasp',default= True, action='store_true', help='Output top-down grasps.')
        parser.add_argument('--debug', default=False,action='store_true', help='Enable debug mode')
        cfgs = parser.parse_args()
        cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        self.side = side
        
        xmin, xmax = -0.19, 0.12
        ymin, ymax = 0.02, 0.15
        zmin, zmax = 0.0, 1.0
        self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    def get_grasp_pose(self,pcd,pcd_mat,cam2base_mat):
        """
        获取抓取姿势
        :param pcd: open3d.geometry.PointCloud, 点云数据
        :param pcd_mat: np.array, 点云变换矩阵
        :param cam2base_mat: np.array
        :return: dict, 抓取姿势
        """
        trans_mat = cam2base_mat @ np.linalg.inv(pcd_mat)
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        args = None
        offset = OFFSET_L if self.side == 'left' else OFFSET_R

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
        if gg is None:
            return args
        elif len(gg) == 0:
            return args
        
        gg = gg.nms().sort_by_score()
        
        '''---------------------从这开始修改-----------------------'''
        gripper_location = [0, 0, 0]  # 假设夹爪位置在原点
        # 提取前20个抓取位姿
        top_20_gg = gg[:20]

        # 定义一个列表来存储抓取位姿及其对应的 offset
        gg_with_offset = []

        # 计算每个抓取位姿的 rot 和 trans，并计算 offset
        for g in top_20_gg:
            # 应用坐标变换
            g_transformed = g.transform(trans_mat)
            
            # 获取旋转矩阵和平移向量
            rot = g_transformed.rotation_matrix
            trans = g_transformed.get_center()
            
            # 计算旋转角度（以度为单位）
            r = R.from_matrix(rot)
            gripper_rot_xyz = r.as_euler('xyz', degrees=True)
            
            # 计算位置偏移（相对于原点的绝对距离）

            loc_offset = abs(trans[0] - gripper_location[0]) + abs(trans[1] - gripper_location[1]) + abs(trans[2] - gripper_location[2])
            
            # 计算旋转偏移（相对于理想旋转的角度差）
            # 假设理想旋转角度为相对于目标位置的方向角

            target_angle = np.arctan2(trans[1] - gripper_location[1], trans[0] - gripper_location[0]) * 180 / np.pi
            # rot_offset = gripper_rot_xyz[2] - np.arctan((trans[1] - gripper_location[1])/(trans[0] - gripper_location[0])) * 180 / np.pi
            rot_offset = gripper_rot_xyz[2] - target_angle
            rot_offset = abs(np.sin(rot_offset / 360 * np.pi))  # 使用sin值确保旋转偏移在0到1之间
            
            # theta = 
            # rot_ele 
            # 计算综合偏移量
            loc_factor = -1.0 # 位置偏移的权重，负的，因为位置越近越好
            rot_factor = 1.0
            offset = loc_factor * loc_offset + rot_factor * rot_offset
            
            # 将抓取位姿及其对应的 offset 添加到列表
            gg_with_offset.append((g, offset))

        # 根据 offset 对抓取位姿从大到小排序
        gg_with_offset.sort(key=lambda x: x[1], reverse=True)
        gg_best = gg_with_offset[0][0]

        '''--------------------------------------------'''
        

        gg_best = gg_best.transform(trans_mat)
        rot = gg_best.rotation_matrix
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        trans = grippers[0].get_center()

        # 获取旋转角度（以度为单位）
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
        args['x'] = trans[0]+offset[0]
        args['y'] = trans[1]+offset[1]
        args['z'] = trans[2]+offset[2]
        args['roll'] = np.deg2rad(gripper_rot_xyz[0])
        args['pitch'] = np.deg2rad(gripper_rot_xyz[1])
        args['yaw'] = np.deg2rad(gripper_rot_xyz[2])
        return args

