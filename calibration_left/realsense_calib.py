import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 本程序realsense_calib.py： 用于给HandEyeCalibration_eye-to-hand_realsense2_left.py提供realsense相机的测试数据

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
       
    def get_camera_intrinsic(self):

        # 获取相机内参
        profile = self.pipeline.get_active_profile()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # 获取内参矩阵
        fx = depth_intrinsics.fx
        fy = depth_intrinsics.fy
        cx = depth_intrinsics.ppx
        cy = depth_intrinsics.ppy
        return fx, fy, cx, cy

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def stop(self):
        self.pipeline.stop()
        

# def get_bounding_box_coordinates(color_image):
#     # 这里调用你的视觉追踪模块，返回bounding box的坐标
#     # 假设返回的坐标格式为 (x, y, width, height)
#     return (100, 100, 50, 50)

# def generate_heatmap(depth_image):
#     heatmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#     return heatmap

# def crop_heatmap(heatmap, bbox):
#     x, y, w, h = bbox
#     cropped_heatmap = heatmap[y:y+h, x:x+w]
#     return cropped_heatmap

if __name__ == '__main__':
    # 初始化相机
    camera = RealSenseCamera()

    try:
        while True:
            depth_image, color_image = camera.get_frames()
            if depth_image is None or color_image is None:
                continue

            # bbox = get_bounding_box_coordinates(color_image)
            # heatmap = generate_heatmap(depth_image)
            # cropped_heatmap = crop_heatmap(heatmap, bbox)

            # 显示结果
            cv2.imshow('RGB Image', color_image)
            cv2.imshow('Heatmap', heatmap)
            cv2.imshow('Cropped Heatmap', cropped_heatmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()