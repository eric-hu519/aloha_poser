#####
#robot controller v2 integrated with anygrasp sdk 
#and deleted any redundant code in v1
#####
from pickle import OBJ
from httpx import get
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
import robot_controller
from utils.robot_utils import depth_circle_sampler, post_process_grasp_pose, check_grasp_pos, args_to_ee_matrix, ee_matrix_to_args, apply_local_offset_to_pose_rpy
from utils.robot_utils import RobotStatusLogger as BotLogger
from scipy.spatial.transform import Rotation as R

#将anygrasp_sdk中的AnyGrasp类导入,添加到依赖路径
import sys
sys.path.append('/home/mamager/interbotix_ws/src/aloha/aloha_poser/anygrasp_sdk/grasp_detection')
from gsnet import AnyGrasp
from point_process_v7 import point_cloud_completed_XY
import open3d as o3d
#from realsense_test_aloha import RealSenseCamera
from utils.pressure_sensor import PressureSensor
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import RealSenseCapture
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import CloudPointProcessor as processor
IMAGE_URL = "http://192.168.31.109:1115/upload"
OFFSET_L = [0.035, -0.1, 0.05, 0.0, 0.17, 0.0]
OFFSET_R = [0.07, 0.057, 0.25]#NOTE: Need recalibrite
DEFAULT_SIDE = 'left'
HOME_POSE = {'left': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0},
             'right': {'x': 0.25, 'y': 0, 'z': 0.350, 'roll': 0, 'pitch': 0, 'yaw': 0}}
GRIPPER_POSE_THRESHOLD = 60
PRESSURE_SENSOR_THRESHOLD = 0.5
OBJ_WIDTH_THRESHOLD = 0.045 #物体宽度阈值
SCALE = 0.01
CAM_ROT = [-56,0,180]
CAM_TRANS = [0,0.45,0.15]
BASE_ROT = [126,0,90]
BASE_TRANS = [0.32,0.31,-0.24]

GRASP_OFFSET = 0.08
RELEASE_OFFSET = -0.05

class Robot_Controller:
    def __init__(self,test_camera = False) -> None:
        """
        :param one_arm_mode: bool, if True, only one arm will be used
        :param test_camera: bool, if True, the realsense camera will not be used, use presaved img instead
        :param enable_pressure: bool, if True, the pressure sensor will be used
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        #init robot
        if DEFAULT_SIDE == 'left':
            self.puppet_bot= InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                        robot_name=f'puppet_left', init_node=True)
            self.puppet_bot.dxl.robot_torque_enable("group", "arm", True)
            self.puppet_bot.dxl.robot_torque_enable("single", "gripper", True)
        else:
            self.puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                        robot_name=f'puppet_right', init_node=True)
            self.puppet_bot.dxl.robot_torque_enable("group", "arm", True)
            self.puppet_bot.dxl.robot_torque_enable("single", "gripper", True)
        
        #init robot status
        self.detect_result = dict()
        self.bot_logger = BotLogger(robot=self.puppet_bot)
        #init camera
        self.test_camera = test_camera
        #init anygrasp sdk
        self.grasp_processor = Anygrasp_Processor(botlogger=self.bot_logger)
        
    def run(self,action_sequence):
        """
        运行动作序列
        """
        #测试模式下直接读取json文件
        if '.json' in action_sequence:
            with open(action_sequence, 'r') as file:
                data = json.load(file)
        else:
            data = action_sequence
        self.logger.info(f"Action length: {len(data)}")
        #init to home pose
        self.home_pose({})
        self.open_gripper({})
        self.run_one_side(data)
        #执行完后归位
        self.sleep_pose({})
        
    def run_one_side(self,action_sequence):
        """
        运行单边动作序列
        """
        #run action sequence
        for item in action_sequence:
            actuator_type = item.get('type')
            name = item.get('name')
            args = item.get('args')
            if args is not None:
                args, detect_target = self.parse_args(args)
            #assign side to operate when detect is not called
            func = self.get_function_by_name(actuator_type, name)
            self.logger.info(f"Running {actuator_type} {name} with args {args} for {DEFAULT_SIDE} arm")
            try:
                # #if not detecting, and grasping, override the rot args for maintaining the grasp angle
                # if args is not None and self.bot_logger.is_grasping:
                #     #use last action to get the rot args
                #     last_action = self.bot_logger.get_last_action()
                #     if last_action is not None:
                #         args['roll'] = last_action.roll
                #         args['pitch'] = last_action.pitch
                #         args['yaw'] = last_action.yaw
                func(args)
                #get current stepid
                stepid = action_sequence.index(item)
                if name == 'grasp':
                    self.bot_logger.is_grasping = True
                elif name == 'release':
                    self.bot_logger.is_grasping = False
                #log robot status
                self.bot_logger.log_action(stepid = stepid)
            except Exception as e:
                self.logger.error(f"Error in {actuator_type} {name} with args {args} for {DEFAULT_SIDE} arm: {e}")
                raise e
        self.home_pose({})

    def init_pose_for_one_side(self):
        """
        单边机械臂初始化
        """
        self.home_pose({})
        self.open_gripper({})

    def parse_args(self, args):
        """
        解析参数，将参数中的detect_result替换为检测结果
        """
        parsed_args = {}
        detect_target = None
        pattern = r'(detect_result)\[["\']?(\w+)["\']?\]\[(\d+)\](\s*[\+\-]?\s*\d+(\.\d+)?(cm)?)?'
        if 'target' in args:
            if args['target'] is not None:
                args['target'].replace(" ","_")
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
                    match.group(2).replace(" ","_")
                    parsed_args[arg] = self.detect_result[match.group(2)][int(match.group(3))] + (SCALE*float(diviation))
                    #parsed_args[arg] = parsed_args[arg][match.group(2)]
                    #parsed_args[arg] = eval(match.group(1),{"detect_result":self.detect_result})[match.group(2)][int(match.group(3))] + (SCALE*float(scale_match.group(1)) if match.group(4) is not None else 0)
                    detect_target = match.group(2)
                else:
                    parsed_args[arg] = np.float32(args[arg])*SCALE
            else:
                parsed_args[arg] = args[arg]
        return parsed_args, detect_target

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

    def get_arm_position(self, get_matrix=True):
        """
        获取机械臂末端位置
        """
        bot = self.puppet_bot
        #get transform matrix
        pose_matrix =  bot.arm.get_ee_pose()
        if get_matrix:
            return pose_matrix
        else:
            ee_pose = pose_matrix[:3,3]
            roll, pitch, yaw = self.get_roll_pitch_yaw_from_matrix(pose_matrix)
        #add roll and pitch to the end of the list
            ee_pose = np.append(ee_pose, [roll, pitch, yaw])
            return ee_pose

    def update_arm_position(self,side):
        """
        获取机械臂末端位置
        depricated
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
    """
    Belows are the APIs to control the robot
    """
    def set_pose(self, args):
        """
        移动机械臂到达指定坐标API
        """
        bot = self.puppet_bot
        bot.arm.set_ee_pose_components(**args)
        
    def restore_vertical(self,args):
        """
        旋转夹爪至竖直位置
        """
        args = {}
        bot = self.puppet_bot
        #get last action
        last_action = self.bot_logger.get_last_action()
        args['x'] = last_action.x
        args['y'] = last_action.y
        args['z'] = last_action.z+0.05
        args['roll'] = 0
        args['pitch'] = 0
        args['yaw'] = 0
        self.set_pose(args)
        
    def restore_horizontal(self,args):
        """
        旋转夹爪至水平位置
        """
        bot = self.puppet_bot
        #get last action
        args = {}
        last_action = self.bot_logger.get_last_action()
        args['x'] = last_action.x
        args['y'] = last_action.y
        args['z'] = last_action.z
        args['roll'] = 0.9
        args['pitch'] = 1.2
        args['yaw'] = 0
        self.set_pose(args)
        
    #抓取目标API
    def grasp(self, args):
        """
        :param args: target coordinates, original Z axis has no offset.
        """
        bot = self.puppet_bot
        args['z'] += GRASP_OFFSET
        #first move above the target
        self.set_pose(args)
        #move down
        args['z'] -= GRASP_OFFSET
        self.set_pose(args)
        #close gripper
        self.close_gripper({})
        #move up
        args['z'] += GRASP_OFFSET
        self.set_pose(args)
    #释放目标API
    def release_obj(self, args):
        """
        :param args: 目标坐标，Z轴有offset
        """
        bot = self.puppet_bot
        #first move above the target
        release_args = {}
        release_args['z'] = RELEASE_OFFSET
        #move down
        self.set_trajectory(release_args)
        #open gripper
        self.open_gripper({})
        #move up
        release_args['z'] = -RELEASE_OFFSET
        self.set_trajectory(release_args)


    def detect(self, args):
        """
        根据指定参数进行目标检测，分配对应机械臂
        """
        color_img = None
        depth_img = None
        mask = None
        x_center = None
        y_center = None
        width = None
        height = None
        combined_pcd = None
        rs_pipline = RealSenseCapture(is_mask=True, use_anchor=True)
        rs_pipline.anchor = None
        #get image from camera
        if self.test_camera:
            #TODO:read color and depth img and mask img from file
            raise NotImplementedError
        else:
            self.push_nlp_to_visiontracker(args['target'])
            color_img, depth_img, mask = rs_pipline.get_color_and_depth_img()
        #mask img by target region
        response = self.send_rgb_img(color_img,IMAGE_URL)
        #get detect result
        response = json.loads(response.content.decode('utf-8'))
        #get object position from tracker
        x_center, y_center, width, height = response['box']
        width += 1
        height += 1
        #convert to 4 point anchor and set anchor mask
        rs_pipline.anchor = np.array([[x_center - width/2, y_center - height/2],[ x_center + width/2, y_center - height/2],[x_center + width/2, y_center + height/2],[x_center - width/2, y_center + height/2]],dtype=np.int32)
        #display the anchor on color image
        #cv2.polylines(color_img, [rs_pipline.anchor], isClosed=True, color=(0, 255, 0), thickness=2)
        #display the color image
        #cv2.imshow('color_img', color_img)
        #cv2.waitKey(3000)
        #cv2.destroyAllWindows()
        #get object position in 3D
        masked_color_img, masked_depth_img = rs_pipline.mask_img(color_img, depth_img)
        #display the masked image
        #cv2.imshow('masked_color_img', masked_color_img)
        #cv2.imshow('masked_depth_img', masked_depth_img)
        #wait for 5 seconds
        #cv2.waitKey(3000)
        #cv2.destroyAllWindows()
        pcd = rs_pipline.get_pcd(masked_color_img, masked_depth_img)
        #SAVE PCD for debug
        o3d.io.write_point_cloud('debug_pcd.ply', pcd)
        #get transform matrix and processed pcd
        _, cam2base_mat = processor.process_pcd(pcd,
                                        rotate_deg=BASE_ROT,
                                        trans_pos=BASE_TRANS
                )
        trans_pcd, pcd_mat = processor.process_pcd(pcd,
                                            rotate_deg=CAM_ROT,
                                            trans_pos=CAM_TRANS,
                                            )
        #for debug save pcd_mat and cam2base_mat
        np.save('debug_pcd_mat.npy', pcd_mat)
        np.save('debug_cam2base_mat.npy', cam2base_mat)
        
        #complete_pcd
        points = np.asarray(trans_pcd.points).astype(np.float32)
        colors = np.asarray(trans_pcd.colors).astype(np.float32)
        o3d.io.write_point_cloud('debug_trans_pcd.ply', trans_pcd)
        # cam_pos = [0,0,0]
        # com_points, com_colors, center, triangle_points = point_cloud_completed_XY(points, colors, cam_pos)
        # trans_pcd.points = o3d.utility.Vector3dVector(com_points)
        # trans_pcd.colors = o3d.utility.Vector3dVector(com_colors)
        #create a plane at max_z
        max_z = np.max(points[:,2])
        #mask points to get better performance
        mean_z = np.mean(points[:,2])
        diff_z = max_z - mean_z
        upper_bound = mean_z + diff_z / 4
        mask = points[:,2] < upper_bound
        points = points[mask]
        colors = colors[mask]
        max_z = np.max(points[:,2])
        trans_pcd = o3d.geometry.PointCloud()
        trans_pcd.points = o3d.utility.Vector3dVector(points)
        trans_pcd.colors = o3d.utility.Vector3dVector(colors)

        #为点云补全工作空间平面
        width = 0.5
        height = 0.5
        plane = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=0.005)
        plane.translate([-width/2, -height/2, max_z+0.00125])
        plane.paint_uniform_color([0.5, 0.5, 0.5])
        #convert the plane to a point cloud
        plane_pcd = plane.sample_points_uniformly(number_of_points=100000)
        #combine current point cloud and the plane
        combined_pcd = trans_pcd + plane_pcd
        #for debug

        o3d.io.write_point_cloud('debug_combined_pcd.ply', combined_pcd)
        #get grasp pose, note offset grasp pose might be None
        grasp_pose, offset_grasp_pose = self.grasp_processor.get_grasp_pose(combined_pcd, pcd_mat, cam2base_mat, self.puppet_bot)
        if grasp_pose is None:
            self.logger.error("Failed to get grasp pose")
            self.sleep_pose({})
            raise Exception("Failed to get grasp pose")
        #update detect result
        self.detect_result[args['target']] = grasp_pose
        self.logger.info(f"Detect result: {self.detect_result[args['target']]}")

    def open_gripper(self,args):
        """
        张开夹爪API
        """
        bot = self.puppet_bot
        #execute open gripper
        bot.gripper.open()

    
    def close_gripper(self,args):
        """
        闭合夹爪API
        """
        bot = self.puppet_bot
        #execute close gripper
        bot.gripper.close(side = DEFAULT_SIDE)


    def home_pose(self,args):
        """
        将机械臂移动到初始位置API
        """
        bot = self.puppet_bot
        bot.arm.set_ee_pose_components(**HOME_POSE[DEFAULT_SIDE])

    def sleep_pose(self,args):
        """
        将机械臂移动到休眠位置API
        """
        bot = self.puppet_bot
        bot.arm.go_to_sleep_pose()
    
    def set_trajectory(self,args):
        """
        以末端为原点进行移动的API
        """
        bot = self.puppet_bot
        bot.arm.set_ee_cartesian_trajectory(**args)
    def set_relative_ee_pose(self, args):
        """
        设置相对末端位置的API
        """
        bot = self.puppet_bot
        bot.arm.set_relative_ee_position_wrt_to_base_frame(**args)
        
    def set_ee_pose_matrix(self, pose_matrix):
        """
        设置末端执行器的变换矩阵
        :param pose_matrix: np.array, 4x4变换矩阵
        """
        bot = self.puppet_bot
        bot.arm.set_ee_pose_matrix(pose_matrix)

class Anygrasp_Processor:
    def __init__(self,side = 'left') -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path',default='/home/mamager/interbotix_ws/src/aloha/aloha_poser/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='Model checkpoint path')
        parser.add_argument('--max_gripper_width', type=float, default=0.18, help='Maximum gripper width (<=0.1m)')
        parser.add_argument('--gripper_height', type=float, default=0.19, help='Gripper height')
        parser.add_argument('--top_down_grasp',default= True, action='store_true', help='Output top-down grasps.')
        parser.add_argument('--debug', default=True,action='store_true', help='Enable debug mode')
        cfgs = parser.parse_args()
        cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        self.side = side
        
        xmin, xmax = -0.19, 0.12
        ymin, ymax = 0.02, 0.15
        zmin, zmax = 0.0, 1.0
        self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    #获取抓取位姿
    def get_grasp_pose(self,pcd,pcd_mat,cam2base_mat,bot=None ):
        """
        :param pcd: open3d.geometry.PointCloud, 点云数据
        :param pcd_mat: np.array, 点云变换矩阵
        :param cam2base_mat: np.array
        :param obj_width: float, 物体宽度, 决定是否采用开合点偏移
        :return: list, 抓取姿势
        """
        trans_mat = cam2base_mat @ np.linalg.inv(pcd_mat)
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        args = None
        offset = OFFSET_L if self.side == 'left' else OFFSET_R

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=False, dense_grasp=False, collision_detection=False)
        if gg is None:
            return args
        elif len(gg) == 0:
            return args
        #check if the grasp is valid, only when bot is not none
        print(f'Total {len(gg)} grasp poses detected')
        grasp_pose_id = 0
        grasp_tobe_removed = []
        gg = gg.nms().sort_by_score()
        offset_grasp_args = None
        for grasp_pose in gg:
            grasp_pose = grasp_pose.transform(trans_mat)
            obj_width = grasp_pose.width
            grasp_args = post_process_grasp_pose(grasp_pose, offset)
            if obj_width >= OBJ_WIDTH_THRESHOLD:
            #if the object width is larger than threshold, use local offset
                offset_grasp_args = apply_local_offset_to_pose_rpy(grasp_args, [0.0,obj_width/2,0.0])
            #convert the grasp matrix to args
            if bot is not None:
                if not check_grasp_pos(bot, grasp_args) or not check_grasp_pos(bot, offset_grasp_args):
                #remove current grasp group by id
                    grasp_tobe_removed.append(grasp_pose_id)
                else:
                    best_pose = grasp_args
                    offset_best_pose = offset_grasp_args
                    break
            grasp_pose_id += 1
        #remove invalid grasp poses
        if len(grasp_tobe_removed) > 0:
            print(f'Removing {len(grasp_tobe_removed)} invalid grasp poses from {len(gg)} total grasp poses')
            gg.remove(grasp_tobe_removed)
        #if no valid grasp pose, return None
        if len(gg) == 0:
            return args
        
        gg_best = gg[0]
        gg_best = gg_best.transform(trans_mat)
        gripper = gg_best.to_open3d_geometry()
        cloud.transform(trans_mat)
        #gripper.transform(trans_mat)
        #gripper.translate(np.array(offset))
        #for debug
        o3d.visualization.draw_geometries([cloud,gripper])
        #NOTE: offset_best_pose might be None
        return best_pose, offset_best_pose


#TODO
## 1. grasp pose calibration programm maybe? DONE
## 2. solve the wired action after grasping DONE
## 3. write a script to test multiple grasp poses DONE
## 4. Multi detect bug? DONE


def main():
    #test the robot controller
    controller = Robot_Controller(test_camera=False)
    controller.run('test_json/one_arm_grasp_move_release.json')
    #controller.run('test_json/one_arm_grasp_multiple_obj.json')
    #controller.run('test_json/one_arm_multiple_detect.json')
    
if __name__ == '__main__':
    main()