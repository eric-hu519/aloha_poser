import numpy as np
import matplotlib.pyplot as plt
import cv2

def image2robot(x,y):
    cam_pose = np.loadtxt('/home/linxing/code/calibration_v2/shanda/param/realsense_camera_pose.txt', delimiter=' ')
    # get_img_tcp.py 通过 TCP 从 RealSense 相机获取图像数据。包括 RGB 图像和深度图，并将相机内参保存在本地文件 D435_intrinsics.txt 中。
    cam_intrinsics = np.loadtxt('/home/linxing/code/calibration_v2/shanda/param/D435_intrinsics.txt', delimiter=' ')
    cam_depth_scale = np.loadtxt('/home/linxing/code/calibration_v2/shanda/param/realsense_camera_depth_scale.txt', delimiter=' ')

    camera_depth_img = cv2.imread('test_image/depth.png', -1)

    click_z = camera_depth_img[y][x] * cam_depth_scale
    print(camera_depth_img[y][x])
    print(click_z)

    click_x = np.multiply(x - cam_intrinsics[0][2], click_z / cam_intrinsics[0][0])
    click_y = np.multiply(y - cam_intrinsics[1][2], click_z / cam_intrinsics[1][1])
    if click_z == 0:
        print('bad depth value!!!')
        return (0,0,0)
    click_point = np.asarray([click_x, click_y, click_z])
    click_point.shape = (3, 1)
    # Convert camera to robot coordinates
    camera2robot = cam_pose
    target_position = np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:]
    target_position = target_position[0:3, 0] # (x,y,z) in robot workspace

    return target_position


if __name__ == '__main__':
    img = cv2.imread('test_image/rgb.png')
    plt.imshow(img)
    plt.show()

    x = int(input('########  x  ########\n'))
    y = int(input('########  y  ########\n'))
    target_position = image2robot(x,y)
    print(target_position)





