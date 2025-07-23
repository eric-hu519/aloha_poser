### ROBOT GRASP CALIBRATION
# This file contains the robot grasp calibration logic.

from robot_controller_v2 import Robot_Controller, Anygrasp_Processor, OFFSET_L, GRASP_OFFSET
import open3d as o3d
import numpy as np
import os

CAM_ROT = [-56,0,180]
CAM_TRANS = [0,0.45,0.15]
BASE_ROT = [126,0,90]
BASE_TRANS = [0.32,0.31,-0.24]

def main():
    #offset = OFFSET_L
    #offset.extend([0.0] * 3)  # Extend offset with three zeros for x, y, z adjustments
    #print(f"Initial offset: {offset}")
    controller = Robot_Controller(test_camera=True)
    args = {}
    grasp_processor = Anygrasp_Processor()
    controller.sleep_pose({})
    #load pcd
    pcd = o3d.io.read_point_cloud(os.path.join('/home/mamager/interbotix_ws/src/aloha/aloha_poser/debug_combined_pcd.ply'))
    #points = np.asarray(pcd.points, dtype=np.float32)
    #colors = np.asarray(pcd.colors, dtype=np.float32)
    o3d.visualization.draw_geometries([pcd])

    #load debug data
    pcd_mat = np.load('debug_pcd_mat.npy')
    cam2base_mat = np.load('debug_cam2base_mat.npy')

    #get grasp pose

    grasp_pose = grasp_processor.get_grasp_pose(pcd, pcd_mat, cam2base_mat, controller.puppet_bot)
    if grasp_pose is None:
        controller.sleep_pose({})
        raise Exception("Failed to get grasp pose")

    args['x'] = grasp_pose[0]
    args['y'] = grasp_pose[1]
    args['z'] = grasp_pose[2] + GRASP_OFFSET
    args['roll'] = grasp_pose[3]
    args['pitch'] = grasp_pose[4]
    args['yaw'] = grasp_pose[5]
    #press y to confirm the pose
    print(f"Grasp pose: {args}")
    print("Press 'y' to confirm the pose, or any other key to exit.")
    key = input()
    if key.lower() != 'y':
        controller.sleep_pose({})
        return
    controller.set_pose(args)
    controller.open_gripper(args)

    preset_offset = np.array(OFFSET_L)
    #extend preset_offset to 6 elements
    #preset_offset = np.append(preset_offset, [0.0, 0.0, 0.0])
    print(f"Preset offset: {preset_offset}")
    offset = np.zeros(6)  # Initialize offset with zeros
    #xy and rotation calibration
    while True:
        org_offset = offset.copy()
        key = input("Please input the key:")
        if key == 'w':
            offset[0] += 0.025
        elif key == 's':
            offset[0] -= 0.025
        elif key == 'a':
            offset[1] += 0.025
        elif key == 'd':
            offset[1] -= 0.025 #right
        elif key == 'q':
            offset[2] += 0.025
        elif key == 'e':
            offset[2] -= 0.025
        elif key == 'r':
            offset[3] += np.deg2rad(5)
        elif key == 'f':
            offset[3] -= np.deg2rad(5)
        elif key == 't':
            offset[4] += np.deg2rad(5)
        elif key == 'g':
            offset[4] -= np.deg2rad(5)
        elif key == 'y':
            offset[5] += np.deg2rad(5)
        elif key == 'h':
            offset[5] -= np.deg2rad(5)
        elif key == 'h':
            controller.close_gripper({})
        elif key == 'j':
            controller.open_gripper({})
        elif key == 'c':
            controller.sleep_pose({})
            break
        elif key == 'x':
            args['z'] -= GRASP_OFFSET
            controller.set_pose(args)
        if not np.array_equal(org_offset, offset):
            args['x'] += offset[0]
            args['y'] += offset[1]
            args['z'] += offset[2]
            args['roll'] += offset[3]
            args['pitch'] += offset[4]
            args['yaw'] += offset[5]
            controller.set_pose(args)
            print(f"Current offset: {offset+ preset_offset}")
if __name__ == "__main__":
    main()