# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import time
import random
import cv2
from interbotix_xs_modules.arm import InterbotixManipulatorXS #导入Interbotix机械臂API
from realsense_test_aloha import RealSenseCamera
import pyrealsense2 as rs


from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

import tqdm
# 实现效果：
#   用于执行右手的手眼标定。该程序使用棋盘格进行标定，设置了采样范围、步长和机械臂的初始位置。
#   程序会遍历采样网格中的每一个点，将机械臂移动到指定位置，并通过相机拍摄图像，检测棋盘格角点。
#   将检测到的棋盘格坐标转化为相机坐标系下的 3D 点，配合机械臂的位姿计算相机-机械臂之间的转换矩阵。
# 使用方法：
# （1）将 UR5Robot 类替换为 Interbotix 机械臂的 API。
# （2）在checkerboard_offset_from_tool中设置棋盘格与末端执行器的偏移量；调整 workspace_limits 为你的机械臂工作范围。
# （3）运行时会提示是否开始标定。采集到数据后，会存储在 calib_result_<timestamp> 文件夹中。最后，通过 SVD 求解相机到基座的刚体变换​



############################# 设置标定相关参数 #############################
cam_type = 'realsense' # or 'realsense'
#checkerboard_offset_from_tool = [-0.002, -0.209, -0.102]  # 设置棋盘格中心点与tcp的位移 #35.1:end-to-board, end-to-tcp:8.75
checkerboard_offset_from_tool = [0.11, 0, 0]  # 设置棋盘格中心点与tcp的位移 #35.1:end-to-board, end-to-tcp:8.75
workspace_limits = np.asarray([
    [-0.04,0.12],  # x 方向范围
    [0.16, 0.45],  # y 方向范围
    [0.3, 0.45]  # z 方向范围
])
# workspace_limits = np.asarray([[-0.206, 0.106], [-0.606, -0.406], [0.106, 0.306]])  # 设置标定时机械臂采样的运动范围
# workspace_limits = np.asarray([[0.25, 0.6], [-0.29, 0.25], [0.06, 0.2]])
# calib_grid_step = 0.05  # 采样步长，决定了三维空间中的采样密度
calib_grid_step = 0.05  # 采样步长
tool_orientation = [0.0, 3.142, 0.0]  # 设置gripper的初始姿态，一般设为垂直于桌面
reset_position = [0.026, -0.529, 0.106]  # 设置机械臂初始位置
#width_min, width_max, height_min, height_max = 0,1280,0,720#250, 1130, 0, 660# 设置桌面采样区域的裁剪范围，用于提升精度
width_min, width_max, height_min, height_max = 0, 640, 0, 480  # 设置桌面采样区域的裁剪范围，用于提升精度
reasonable_depth_range = [300, 990]  # 根据物理测量，给深度值设置合理的分布区间，在进行实际采样时可以用来过滤异常采样点
test_calib_result = False#True #是否对标定结果进行误差测试
###########################################################################

camera = RealSenseCamera()#实例化相机对象

# 获取相机内参
# 获取相机内参
fx, fy, cx, cy = camera.get_camera_intrinsic()

# 创建文件夹，保存标定结果
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
if not os.path.exists('calib_result_' + str(curr_time) + '/'):
    os.makedirs('calib_result_' + str(curr_time) + '/')
if not os.path.exists('calib_result_' + str(curr_time) + '/capture_image/'):
    os.makedirs('calib_result_' + str(curr_time) + '/capture_image/valid/')
    os.makedirs('calib_result_' + str(curr_time) + '/capture_image/not_detected/')
    os.makedirs('calib_result_' + str(curr_time) + '/capture_image/bad_depth/')
    img_save_path_valid = 'calib_result_' + str(curr_time) + '/capture_image/valid/'
    img_save_path_nondetect = 'calib_result_' + str(curr_time) + '/capture_image/not_detected/'
    img_save_path_baddepth = 'calib_result_' + str(curr_time) + '/capture_image/bad_depth/'
f_log = open('calib_result_' + str(curr_time) + '/log.txt', 'a')
'''
记录所有采样数据，格式如下：
[u v],[xc yc zc],[xw yw zw],[valid](or [invalid,checkboard not detected] or [invalid,checkboard detected but depth is bad])
'''
f_sample_point = open('calib_result_' + str(curr_time) + '/data_record.txt', 'a')
f_sample_point.write('----- Data format: [u v],[xc yc zc],[xw yw zw],[valid] -----\n')

# Construct 3D calibration grid across workspace
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1],
                          int(1.0 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1],
                          int(1.0 + (workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], int(1.0 + (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)) #np.array([0.327,0.347,0.367])#
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2] #返回数值是三维空间离散点的总数目N
calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1) #最终转化为(N,3)的矩阵，每一行保存的是采样点的(x,y,z)真实世界坐标
# print("calib_grid_pts",calib_grid_pts)

measured_pts = []
observed_pts = []
observed_pix = []

# Move robot to home pose
print('Connecting to robot...')
#   实例化机械臂
#robot = UR5Robot(tcp_host_ip, vel=0.06, acc=0.5, camera_type=cam_type, workspace_limits=workspace_limits)
# 左从臂：vx300s
puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_right', init_node=True)
print('Please check the chessboard...')
time.sleep(1.5)
#   机械臂归位
puppet_bot_right.arm.set_ee_pose_components(x=0.2, z=0.4)
#robot.move_to_position(reset_position, tool_orientation)
# startNow = input('########  Start calibration (y/n)?  ########\n')
# if startNow == 'y':
#     pass
# elif startNow == 'n':
#     exit(0)

# Move robot to each calibration point in workspace
print('Start collecting data...')
# puppet_bot_left.arm.set_single_joint_position("waist", np.pi/2.0)
frame_id = 0
for calib_pt_idx in tqdm.tqdm(range(num_calib_grid_pts)):
 #遍历所有的采样点
    tool_position = calib_grid_pts[calib_pt_idx, :] #得到当前点(x,y,z)坐标
    frame_id += 1
    # puppet_bot_left.arm.set_ee_pose_components(x=0.3, z=0.2, roll=0.0, pitch=0, yaw=0.0)
    #puppet_bot_left.arm.set_ee_cartesian_trajectory(x=0.2, z=0.4, roll=0.0, pitch=0, yaw=0.0)
    puppet_bot_right.arm.set_ee_pose_components(x=tool_position[0], y=tool_position[1], z=tool_position[2], pitch=-0.1,)
    #robot.move_to_position(tool_position, tool_orientation) #控制机械臂末端tcp(too center point)移动到指定点
    #print('Robot moves to:', tool_position, tool_orientation)
    print('Robot moves to:', tool_position)
    print(f"当前采样为第{frame_id}帧")
    # 获取当前机械臂末端执行器的位姿
    ee_pose = puppet_bot_right.arm.get_ee_pose()
    # 从变换矩阵中提取三维坐标
    ee_position = ee_pose[:3, 3]
    #puppet_bot_left.arm.set_ee_pose_components(x=0.35, y=-0.35, z=0.35)

    print("当前机械臂夹爪的三维坐标为：", ee_position)
    
    time.sleep(2)
    # Find checkerboard center 棋盘格尺寸12*9
    checkerboard_size = (3, 3)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # get camera data
    try:
        camera_depth_img, camera_color_img = camera.get_frames()
        #camera_color_img, camera_depth_img = robot.get_realsense_sensor_data()
        print("get img")
    except Exception as e:
        print("Error getting camera data:", e)
        # import ipdb;

        # ipdb.set_trace()
        # camera_color_img = robot.get_kinect_camera_data_rgb()
        # camera_depth_img = robot.get_kinect_camera_data_depth()

    cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'rgb.png', camera_color_img)
    cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'depth.png', camera_depth_img)
    print("save img")
    print('Data capture done!')
    bgr_color_data = camera_color_img
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_BGR2GRAY)
    # 用原图(1920*1080)进行棋盘格检测时，由于在整幅图中棋盘格很小，所以检测不出来的概率很大。
    # 所以这里采用一种先裁剪ROI再还原的方式来提高棋盘格检测成功率。
    gray_data = gray_data[height_min:height_max, width_min:width_max]
    cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'gray.png', gray_data)

    checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH) #获取棋盘格中心点
    # 修改中心点索引

    if checkerboard_found: #如果检测到棋盘格中心点
        corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria) #进一步细化，得到棋盘格中心点的精确图像坐标
        corners_refined = corners
        corners[4, 0, :][0] = corners[4, 0, :][0] + width_min
        corners[4, 0, :][1] = corners[4, 0, :][1] + height_min
        # Get observed checkerboard center 3D point in camera space
        checkerboard_pix = np.round(corners[4, 0, :]).astype(int) #四舍五入保留
        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]] #获取棋盘中心点对应的相机深度值
        # 将棋盘中心点的像素值坐标(u,v)化为相机坐标系下的三维相机坐标点(x_c,y_c,z_c),其中z_c直接从深度相机获取，单位mm
        # 获取相机内参
        
        checkerboard_x = np.multiply(checkerboard_pix[0] - cx, checkerboard_z / fx)
        checkerboard_y = np.multiply(checkerboard_pix[1] - cy, checkerboard_z / fy)
        # checkerboard_x = np.multiply(checkerboard_pix[0] - robot.cam_intrinsics[0][2],  #相机的内参矩阵在哪里获取的？
        #                              checkerboard_z / robot.cam_intrinsics[0][0])
        # checkerboard_y = np.multiply(checkerboard_pix[1] - robot.cam_intrinsics[1][2],
        #                              checkerboard_z / robot.cam_intrinsics[1][1])
        if checkerboard_z < reasonable_depth_range[0] or checkerboard_z > reasonable_depth_range[1]:  # checkerboard_z == 0:
            ## 如果当前点的深度值是坏点，直接跳过该点
            num_baddepth = len(os.listdir(img_save_path_baddepth))
            cv2.imwrite(img_save_path_baddepth + str(num_baddepth) + '.png', bgr_color_data)
            # [u v],[xc yc zc],[xw yw zw],[valid](or [invalid,checkboard not detected] or [invalid,checkboard detected but depth is bad])
            tool_position_tmp = tool_position + checkerboard_offset_from_tool
            f_sample_point.write(
                '[baddepth_' + str(num_baddepth) + '.png],' + '[' + str(checkerboard_pix[0]) + ' ' + str(
                    checkerboard_pix[1]) + '],[' + str(checkerboard_x) + ' ' + str(checkerboard_y) + ' ' + str(
                    checkerboard_z) + '],[' + str(tool_position_tmp[0]) + ' ' + str(tool_position_tmp[1]) + ' ' + str(
                    tool_position_tmp[2]) + '],[invalid,checkboard detected but depth is bad]' + '\n')
            continue

        # Save calibration point and observed checkerboard center
        observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z]) #保存观测点(棋盘格中心点)在相机坐标系下的坐标(x_c,y_c,z_c)
        tool_position = tool_position + checkerboard_offset_from_tool

        measured_pts.append(tool_position) #保存观测点(棋盘格中心点)在基坐标系下的坐标(x_w,y_w,z_w)
        observed_pix.append(checkerboard_pix) #保存观测点(棋盘格中心点)的像素坐标(u,v)

        # Draw and display the corners
        vis = cv2.drawChessboardCorners(bgr_color_data, (1, 1), corners[4, :, :], checkerboard_found)
        cv2.imwrite(img_save_path_valid + '%06d.png' % len(measured_pts), bgr_color_data)
        f_sample_point.write(
            '[valid_' + str(len(measured_pts)) + '.png],' + '[' + str(checkerboard_pix[0]) + ' ' + str(
                checkerboard_pix[1]) + '],[' + str(checkerboard_x) + ' ' + str(checkerboard_y) + ' ' + str(
                checkerboard_z) + '],[' + str(tool_position[0]) + ' ' + str(tool_position[1]) + ' ' + str(
                tool_position[2]) + '],[valid]' + '\n')
    else:
        num_nondetect = len(os.listdir(img_save_path_nondetect))
        cv2.imwrite(img_save_path_nondetect + str(num_nondetect) + '.png', bgr_color_data)
        tool_position_tmp = tool_position + checkerboard_offset_from_tool
        f_sample_point.write(
            '[nondetect_' + str(num_nondetect) + '.png],' + '[none none none],[none none none],[' + str(tool_position_tmp[0]) + ' ' + str(tool_position_tmp[1]) + ' ' + str(
                tool_position_tmp[2]) + '],[invalid,checkboard not detected]' + '\n')

f_sample_point.close() #采样结束，保存全部采样点
f_log.write('Device: '+cam_type+'\n'+'Pre-defined sample points: '+str(num_calib_grid_pts)+'\n'+'Valid sample point: '+str(len(os.listdir(img_save_path_valid)))+'\n'+'Invalid sample (bad depth): '+str(len(os.listdir(img_save_path_baddepth)))+'\n'+'Invalid sample (not detect): '+str(len(os.listdir(img_save_path_nondetect)))+'\n')
# robot.move_to_position(reset_position, tool_orientation) #采样完成，机械臂归位
puppet_bot_right.arm.go_to_home_pose()
'''
T_world2cam表示相机坐标系和基坐标系之间转换矩阵,要求解的就是这个
np.eye(4)=[1 0 0 0
           0 1 0 0
           0 0 1 0
           0 0 0 1]
'''
# 存储采样点，进行标定计算
measured_pts = np.asarray(measured_pts) #基坐标系下的坐标(x_w,y_w,z_w)
observed_pts = np.asarray(observed_pts) #相机坐标系下的坐标(x_c,y_c,z_c)
observed_pix = np.asarray(observed_pix) #像素坐标(u,v)
world2camera = np.eye(4)

# Estimate rigid transform with SVD (from Nghia Ho)
'''
目标：求解相机坐标系和基坐标系之间转换矩阵
Note: 这里并没有根据A*X=X*B的标定矩阵来求解，而是参考了Nghia Ho的求解方法，具体如下:
通过相机内参以及机械臂的采集数据，我们可以获取若干位姿点，经过转换后可以得到两个点集，
A{Xw,Yw,Zw}和B{Xc,Yc,Zc},其中A为基坐标系下的点，B为相机坐标系下的点。然后通过SVD方法
直接求解转换矩阵[R T].
Reference: http://nghiaho.com/?page_id=671
'''
def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]  # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # Special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t

'''
通过调用get_rigid_transform()已经可以求出手眼标定矩阵，但实际应用中相机(Realsense,Kinect)存在噪声误差，
采集到的深度值可能不准。这里其实就是通过优化算法来近似拟合出一个相机深度值矫正系数(z_scale)。具体来说，
一开始求解手眼标定矩阵时，我们将z_scale默认设为1。然后将z_scale置为一个变量，再重新计算出新的标定矩阵。
计算新旧标定矩阵之间的均方差，通过优化算法(e.g. optimize.minimize)使该均方差最小，相应得到的z_scale即
为合适的深度值矫正系数。
'''
def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, world2camera, new_observed_pts

    # Apply z offset and compute new observed points using camera intrinsics
    observed_z = observed_pts[:, 2:] * z_scale
    print("observed_z",observed_z)
    fx, fy, cx, cy = camera.get_camera_intrinsic()
    observed_x = np.multiply(observed_pix[:, [0]] - cx, observed_z / fx)
    observed_y = np.multiply(observed_pix[:, [1]] - cy, observed_z / fy)
    # observed_x = np.multiply(observed_pix[:, [0]] - robot.cam_intrinsics[0][2], observed_z / robot.cam_intrinsics[0][0])
    # observed_y = np.multiply(observed_pix[:, [1]] - robot.cam_intrinsics[1][2], observed_z / robot.cam_intrinsics[1][1])
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points and new observed points
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
    t.shape = (3, 1)
    world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

    # Compute rigid transform error
    registered_pts = np.dot(R, np.transpose(measured_pts)) + np.tile(t, (1, measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error / measured_pts.shape[0])
    print('rmse: ', rmse)
    return rmse


# Optimize z scale w.r.t. rigid transform error
print('Start calibrating...')
z_scale_init = 1
optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
camera_depth_offset = optim_result.x

# Save camera optimized offset and camera pose
print('Saving calibration result...')
# robot.open_gripper()
np.savetxt('calib_result_' + str(curr_time) + '/realsense_camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
get_rigid_transform_error(camera_depth_offset)
camera_pose = np.linalg.inv(world2camera)
np.savetxt('calib_result_' + str(curr_time) + '/realsense_camera_pose.txt', camera_pose, delimiter=' ')
print('Calibration finish!')


'''
###################### Test #########################
#标定完成之后，进行随机采样收集位姿点，然后进行误差计算#
#####################################################
'''
if test_calib_result:
    # workspace_limits = np.asarray([[0.366, 0.536], [-0.196, 0.136], [0.106, 0.306]])
    test_num = 40
    test_xw_list = np.array(random.sample(list(range(int(workspace_limits[0][0]*1000), int(workspace_limits[0][1]*1000))),test_num)) / 1000.0
    test_yw_list = np.array(random.sample(list(range(int(workspace_limits[1][0]*1000), int(workspace_limits[1][1]*1000))), test_num)) / 1000.0
    test_zw_list = np.array(random.sample(list(range(int(workspace_limits[2][0]*1000), int(workspace_limits[2][1]*1000))), test_num)) / 1000.0

    os.makedirs('calib_result_' + str(curr_time) + '/capture_image_test/valid/')
    os.makedirs('calib_result_' + str(curr_time) + '/capture_image_test/not_detected/')
    os.makedirs('calib_result_' + str(curr_time) + '/capture_image_test/bad_depth/')
    img_save_path_valid_test = 'calib_result_' + str(curr_time) + '/capture_image_test/valid/'
    img_save_path_nondetect_test = 'calib_result_' + str(curr_time) + '/capture_image_test/not_detected/'
    img_save_path_baddepth_test = 'calib_result_' + str(curr_time) + '/capture_image_test/bad_depth/'

    test_measured_pts = []
    test_observed_pts = []
    test_observed_pix = []
    # Move robot to each calibration point in workspace
    print('Start collecting test data...')
    time.sleep(3)
    for calib_pt_idx in range(test_num):
        tool_position = np.array([test_xw_list[calib_pt_idx], test_yw_list[calib_pt_idx], test_zw_list[calib_pt_idx]]) #calib_grid_pts[calib_pt_idx, :]
        puppet_bot_right.arm.go_to_home_pose()
        # robot.move_to_position(tool_position, tool_orientation)
        print('Robot moves to:', tool_position, tool_orientation)
        # Find checkerboard center
        checkerboard_size = (3, 3)
        refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # get camera data
        try:
            # camera_color_img, camera_depth_img = robot.get_realsense_sensor_data()
            camera_color_img, camera_depth_img = camera.get_frames()
            print("get img")
        except:
            import ipdb;

            ipdb.set_trace()
            camera_color_img = robot.get_kinect_camera_data_rgb()
            camera_depth_img = robot.get_kinect_camera_data_depth()

        cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'rgb_test.png', camera_color_img)
        cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'depth_test.png', camera_depth_img)
        print('Data capture done!')
        bgr_color_data = camera_color_img
        gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_BGR2GRAY)
        # 用原图(1920*1080)进行棋盘格检测时，由于在整幅图中棋盘格很小，所以检测不出来的概率很大。
        # 所以这里采用一种先裁剪ROI再还原的方式来提高棋盘格检测成功率。
        gray_data = gray_data[height_min:height_max, width_min:width_max]
        cv2.imwrite('calib_result_' + str(curr_time) + '/' + 'gray_test.png', gray_data)

        checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                                cv2.CALIB_CB_ADAPTIVE_THRESH)
        if checkerboard_found:
            corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)
            corners_refined = corners
            corners[4, 0, :][0] = corners[4, 0, :][0] + width_min
            corners[4, 0, :][1] = corners[4, 0, :][1] + height_min
            # Get observed checkerboard center 3D point in camera space
            checkerboard_pix = np.round(corners[4, 0, :]).astype(int)
            checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]] * camera_depth_offset
            fx, fy, cx, cy = camera.get_camera_intrinsic()
            checkerboard_x = np.multiply(checkerboard_pix[0] - cx, checkerboard_z / fx)
            checkerboard_y = np.multiply(checkerboard_pix[1] - cy, checkerboard_z / fy)
            # checkerboard_x = np.multiply(checkerboard_pix[0] - robot.cam_intrinsics[0][2],
            #                              checkerboard_z / robot.cam_intrinsics[0][0])
            # checkerboard_y = np.multiply(checkerboard_pix[1] - robot.cam_intrinsics[1][2],
            #                              checkerboard_z / robot.cam_intrinsics[1][1])
            if checkerboard_z < reasonable_depth_range[0] or checkerboard_z > reasonable_depth_range[1]:  # checkerboard_z == 0:
                num_baddepth = len(os.listdir(img_save_path_baddepth_test))
                cv2.imwrite(img_save_path_baddepth_test + str(num_baddepth) + '.png', bgr_color_data)
                continue

            # Save calibration point and observed checkerboard center
            test_observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
            tool_position = tool_position + checkerboard_offset_from_tool
            test_measured_pts.append(tool_position)
            test_observed_pix.append(checkerboard_pix)

            # Draw and display the corners
            vis = cv2.drawChessboardCorners(bgr_color_data, (1, 1), corners[4, :, :], checkerboard_found)
            cv2.imwrite(img_save_path_valid_test + '%06d.png' % len(test_measured_pts), bgr_color_data)
        else:
            num_nondetect = len(os.listdir(img_save_path_nondetect_test))
            cv2.imwrite(img_save_path_nondetect_test + str(num_nondetect) + '.png', bgr_color_data)

    # Estimate rigid transform between measured points and new observed points
    R = world2camera[0:3, 0:3]
    t = world2camera[0:3, -1]
    t.shape = (3,1)
    # Compute rigid transform error
    print(R.shape)
    print(test_measured_pts)
    print((np.transpose(test_measured_pts)).shape)
    eval_registered_pts = np.dot(R, np.transpose(test_measured_pts)) + np.tile(t, (1, test_measured_pts.shape[0]))
    error = np.transpose(eval_registered_pts) - test_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error/test_measured_pts.shape[0])

    # eval_registered_pts = np.dot(R, np.transpose(test_measured_pts)) + np.tile(t, (1, test_measured_pts.shape[0]))
    # error = np.transpose(eval_registered_pts) - test_observed_pts
    # error = np.sum(np.multiply(error, error))
    # rmse = np.sqrt(error / test_measured_pts.shape[0])

    print("--------------- The RMSE is:", rmse, ' ----------------')
    if rmse < 1e-5:
        print("Everything looks good!")
    else:
        print("Hmm something doesn't look right ...")

    ## Draw figure to show the error ##
    # (x1,y1,z1)表示真实观察值
    x1 = test_observed_pts[:, 0]
    y1 = test_observed_pts[:, 1]
    z1 = test_observed_pts[:, 2]
    # (x2,y2,z2)表示通过转换矩阵换算出的估计值
    x2 = np.transpose(eval_registered_pts)[:, 0]
    y2 = np.transpose(eval_registered_pts)[:, 1]
    z2 = np.transpose(eval_registered_pts)[:, 2]
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, c='r', label='真实值')
    ax.scatter(x2, y2, z2, c='g', label='估计值')
    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # 展示
    plt.show()
