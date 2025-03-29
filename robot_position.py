import numpy as np
from robot_controller import robot_controller
from sympy import true
import time
import json
import os
def main():
    args={}
    side = 'left'
    args['target'] = "red_straw"
    controller = robot_controller(test_camera=True)
    #detect_pose = controller.detect(args, 'left')
    # target_pos = {}
    # target_pos["x"]=float(detect_pose[0])
    # target_pos["y"]=float(detect_pose[1])
    # target_pos["z"]=float(detect_pose[2])
    init_pose = {}
    init_pose["x"] = float(0.25)
    init_pose["y"] = float(0)
    init_pose["z"] = float(0.35)
    init_pose["roll"] = np.deg2rad(0)
    init_pose["pitch"] = np.deg2rad(0)
    init_pose["yaw"] = np.deg2rad(0)
    controller.set_pose(init_pose, side)
    controller.open_gripper(init_pose,side=side)
    controller.close_gripper(init_pose,side=side)
    
    init_pose["x"] = 0.17629263+0.07
    init_pose["y"] = -0.30080539+0.14
    init_pose["z"] = -0.082287+0.25
    init_pose["roll"] = np.deg2rad(0)
    init_pose["pitch"] = np.deg2rad(79.118520)
    init_pose["yaw"] = np.deg2rad(0)
    controller.set_pose(init_pose, side)
    controller.open_gripper(init_pose,side=side)
    #use keyboard to adjust the position
    #w will add x, s will minus x, a will add y, d will minus y, q will add z, e will minus z
    while True:
        key = input("Please input the key:")
        org_pose = init_pose.copy()
        if key == 'w':
            init_pose["x"] += 0.01
        elif key == 's':
            init_pose["x"] -= 0.01
        elif key == 'a':
            init_pose["y"] += 0.01
        elif key == 'd':
            init_pose["y"] -= 0.01
        elif key == 'q':
            init_pose["z"] += 0.01
        elif key == 'e':
            init_pose["z"] -= 0.01
        elif key == 'r':
            init_pose["roll"] += 0.1
        elif key == 'f':
            init_pose["roll"] -= 0.1
        elif key == 't':
            init_pose["pitch"] += 0.1
        elif key == 'g':
            init_pose["pitch"] -= 0.1
        elif key == 'y':
            init_pose["yaw"] += 0.1
        elif key == 'h':
            controller.close_gripper(init_pose,side=side)
        elif key == 'j':
            controller.open_gripper(init_pose,side=side)
        elif key == 'c':
            controller.sleep_pose({}, side)
            break
        if org_pose != init_pose:
            controller.set_pose(init_pose, side)
            print(f"current position: {init_pose}")
    #save current position and target position to json
    #add current pair to json list
    # if not os.path.exists('position.json'):
    #     with open('position.json', 'w') as file:
    #         new_data = [{"robot_pose":init_pose, "target":target_pos, "side":"left"}]
    #         json.dump(new_data, file)
    # else:
    #     with open('position.json', 'r') as file:
    #         data = json.load(file)
    #     data.append({"robot_pose":init_pose, "target":target_pos, "side":"left"})
    #     #save to json
    #     with open('position.json', 'w') as file:
    #         json.dump(data, file)
    # print("Position saved")

if __name__ == '__main__':
    main()