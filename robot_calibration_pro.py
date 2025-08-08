#在采集工作空间点云后，对五个校准点位进行拟合，实现机械臂校准

from robot_controller_v2 import Robot_Controller, Anygrasp_Processor
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import RealSenseCapture
from anygrasp_sdk.grasp_detection.cloud_point_process_v2 import CloudPointProcessor as processor
import numpy as np
import open3d as o3d
CAM_ROT = [-56,0,180]
CAM_TRANS = [0,0.45,0.15]
BASE_ROT = [126,0,90]
BASE_TRANS = [0.32,0.31,-0.24]
def interactive_gripper_adjustment(cloud, gripper):
    """
    Interactive gripper position adjustment using keyboard controls.
    """
    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Gripper Position Adjustment", 1280, 720)
    
    # 添加几何体
    vis.add_geometry(cloud)
    vis.add_geometry(gripper)
    
    # 设置移动步长
    step_size = 0.01  # 1cm
    
    # 定义键盘回调函数
    def move_gripper_x_positive(vis):
        gripper.translate([step_size, 0, 0])
        vis.update_geometry(gripper)
        print(f"Gripper moved +X by {step_size}")
        return False
    
    def move_gripper_x_negative(vis):
        gripper.translate([-step_size, 0, 0])
        vis.update_geometry(gripper)
        print(f"Gripper moved -X by {step_size}")
        return False
    
    def move_gripper_y_positive(vis):
        gripper.translate([0, step_size, 0])
        vis.update_geometry(gripper)
        print(f"Gripper moved +Y by {step_size}")
        return False
    
    def move_gripper_y_negative(vis):
        gripper.translate([0, -step_size, 0])
        vis.update_geometry(gripper)
        print(f"Gripper moved -Y by {step_size}")
        return False
    
    def move_gripper_z_positive(vis):
        gripper.translate([0, 0, step_size])
        vis.update_geometry(gripper)
        print(f"Gripper moved +Z by {step_size}")
        return False
    
    def move_gripper_z_negative(vis):
        gripper.translate([0, 0, -step_size])
        vis.update_geometry(gripper)
        print(f"Gripper moved -Z by {step_size}")
        return False
    
    def increase_step_size(vis):
        nonlocal step_size
        step_size *= 2
        print(f"Step size increased to {step_size}")
        return False
    
    def decrease_step_size(vis):
        nonlocal step_size
        step_size /= 2
        print(f"Step size decreased to {step_size}")
        return False
    
    def print_gripper_position(vis):
        center = gripper.get_center()
        print(f"Current gripper position: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
        return False
    
    def save_position(vis):
        center = gripper.get_center()
        print(f"Position saved: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
        # 这里可以保存到文件或返回位置信息
        return False
    
    # 注册键盘回调 (使用ASCII码)
    vis.register_key_callback(ord("D"), move_gripper_x_positive)  # D键: +X
    vis.register_key_callback(ord("A"), move_gripper_x_negative)  # A键: -X
    vis.register_key_callback(ord("W"), move_gripper_y_positive)  # W键: +Y
    vis.register_key_callback(ord("S"), move_gripper_y_negative)  # S键: -Y
    vis.register_key_callback(ord("Q"), move_gripper_z_positive)  # Q键: +Z
    vis.register_key_callback(ord("E"), move_gripper_z_negative)  # E键: -Z
    vis.register_key_callback(ord("="), increase_step_size)       # =键: 增加步长
    vis.register_key_callback(ord("-"), decrease_step_size)       # -键: 减少步长
    vis.register_key_callback(ord("P"), print_gripper_position)   # P键: 打印位置
    vis.register_key_callback(ord(" "), save_position)            # 空格键: 保存位置
    
    # 打印控制说明
    print("=== Gripper Position Control ===")
    print("W/S: Move gripper forward/backward (Y axis)")
    print("A/D: Move gripper left/right (X axis)")
    print("Q/E: Move gripper up/down (Z axis)")
    print("+/-: Increase/decrease step size")
    print("P: Print current position")
    print("Space: Save current position")
    print("ESC: Exit")
    print(f"Current step size: {step_size}")
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()
    
    # 返回最终的gripper位置
    final_position = gripper.get_center()
    return final_position, gripper

def capture_cpd():
    """
    Capture point cloud data for calibration.
    """
    # Initialize the RealSense camera
    camera = RealSenseCapture(is_mask=True, use_anchor=False)
    color_img, depth_img, mask = camera.get_color_and_depth_img()
    pcd = camera.get_pcd(color_img, depth_img)

        #get transform matrix and processed pcd
    _, cam2base_mat = processor.process_pcd(pcd,
                                    rotate_deg=BASE_ROT,
                                    trans_pos=BASE_TRANS
            )
    trans_pcd, pcd_mat = processor.process_pcd(pcd,
                                        rotate_deg=CAM_ROT,
                                        trans_pos=CAM_TRANS,
                                        )
    #add gripper at origin

    o3d.visualization.draw_geometries([trans_pcd])

    return trans_pcd, pcd_mat, cam2base_mat

def grasp_pose_generator(pcd, pcd_mat, cam2base_mat):
    """
    Generate grasp positions from the point cloud data.
    """
    grasp_processor = Anygrasp_Processor()
    trans_mat = cam2base_mat @ np.linalg.inv(pcd_mat)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)

    gg, cloud = grasp_processor.anygrasp.get_grasp(points, colors, lims = grasp_processor.lims,apply_object_mask=False, dense_grasp=False, collision_detection=False)
    if gg is None:
        print("No grasp found")
        return None
    gg = gg.nms().sort_by_score()
    # gg_pick = gg[]
    # gripper_list = []
    # for pose in gg_pick:
    #     pose.transform(trans_mat)
    #     gripper_list.append(pose.to_open3d_geometry())
    gg_best = gg[0]
    gg_best = gg_best.transform(trans_mat)
    gripper = gg_best.to_open3d_geometry()
    cloud.transform(trans_mat)
    #o3d.visualization.draw_geometries([cloud, gripper])
    final_position, adjusted_gripper = interactive_gripper_adjustment(cloud, gripper)
    print(f"Final gripper position: {final_position}")
    return cloud, adjusted_gripper




def main():
    
    # Capture point cloud data
    trans_pcd, pcd_mat, cam2base_mat = capture_cpd()
    grasp_pose_generator(trans_pcd, pcd_mat, cam2base_mat)

if __name__ == "__main__":
    main()