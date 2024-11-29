import cv2
import numpy as np

# 读取图像并转换为灰度图像
image_path = '/home/mamager/interbotix_ws/src/aloha/act-plus-plus/aloha_poser/calibration/calib_result_20230816212252/capture_image/valid/000004.png'  # 替换为你的图像路径
color_image = cv2.imread(image_path)
gray_data = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 棋盘格尺寸
checkerboard_size = (2, 2)  # 替换为你的实际棋盘格尺寸

# 检测棋盘格角点
checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)

if checkerboard_found:
    # 进一步细化角点
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)
    
    # 打印角点信息
    print("Checkerboard corners refined:", corners_refined)
    
    # 获取棋盘格中心点
    checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
    print("Checkerboard center pixel coordinates:", checkerboard_pix)
else:
    print("Checkerboard not found")

# 显示图像和检测结果
cv2.drawChessboardCorners(color_image, checkerboard_size, corners, checkerboard_found)
cv2.imshow('Checkerboard Detection', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()