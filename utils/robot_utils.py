import numpy as np
def depth_circle_sampler(radius,depth_img, x_center,y_center):
    """
    sample a circle around x_center and y_center and return the average depth value in the circle
    """
    x = np.arange(depth_img.shape[1])
    y = np.arange(depth_img.shape[0])
    xx, yy = np.meshgrid(x, y)
    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2 < radius ** 2
    circle_depth = depth_img[circle]
    return np.mean(circle_depth)