import numpy as np
import matplotlib.pyplot as plt
import time
import open3d as o3d

from numba import njit
from shapely.affinity import rotate
from shapely.geometry import MultiPoint, Point, Polygon
from collections import defaultdict
from shapely import speedups
from matplotlib.path import Path

def compute_2d_min_obb(points_2d):
    """
    计算二维点云的最小旋转包围盒 (OBB) 以及几何中心

    参数:
        points_2d: (N, 2) 的 NumPy 数组，二维点云

    返回:
        box_coords: (4, 2) OBB 四个顶点坐标
        center: (2,) 包围盒中心点坐标
    """
    if not isinstance(points_2d, np.ndarray):
        points_2d = np.array(points_2d)
        
    if points_2d.shape[1] != 2:
        raise ValueError("输入必须是二维点云，每个点形如 (x, y)")

    # 创建 shapely 的多点对象
    multipoint = MultiPoint(points_2d)

    # 获得最小旋转矩形
    obb = multipoint.minimum_rotated_rectangle

    # 获取 OBB 顶点
    x, y = obb.exterior.coords.xy
    box_coords = np.array(list(zip(x, y)))[:-1]  # 最后一个点等于第一个，去掉重复

    # 计算 OBB 中心（4 个顶点平均）
    center = np.mean(box_coords, axis=0)

    return box_coords, center



def project_to_xy(points_4d):
    """
    将四维点云的前三维投影到 XY 平面，返回二维坐标。

    参数：
        points_4d: (N, 4) 点云数组

    返回：
        xy: (N, 2) 数组，含 (x, y)
    """
    xy = points_4d[:, :2]
    return xy


def project_point_cloud_to_plane(points, plane_normal, plane_point): 
    """
    将四维点云的前三维投影到指定平面，最终输出为二维点云集。

    参数：
        points: (N, 4) 的点云数组，前三维为坐标，第四维为属性。
        plane_normal: 平面法向量 (3,) 数组
        plane_point: 平面上一点 (3,) 数组

    返回：
        projected_xy: (N, 2) 数组，含投影到指定平面的二维坐标
    """
    points = np.asarray(points)
    coords = points[:, :3]       # 提取前三维坐标

    normal = np.asarray(plane_normal)
    normal = normal / np.linalg.norm(normal)  # 单位化法向量
    plane_point = np.asarray(plane_point)

    # 向量从平面点指向点
    vecs = coords - plane_point
    distances = np.dot(vecs, normal)  # 点到平面的距离（有向）
    projected_coords = coords - np.outer(distances, normal)  # 投影公式

    # 提取投影坐标中的XY部分
    projected_xy = projected_coords[:, 0:2]

    return projected_xy


def get_half_height(points):
    """
    输入: points (N, 4) 的三维点云
    返回: 点云高度的一半（z轴最大值-最小值的一半）
    """
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    height = z_max - z_min
    return height / 2


def mirror_point_cloud_over_line(points, line_point, line_vector): 
    """
    将四维点云（前三维为空间坐标，第四维为属性）沿z轴分层，
    对每层点的xy坐标关于指定二维直线进行镜像操作，
    并保持z和第四维属性不变，最后与原点云合并返回。

    参数：
        points: (N, 4) 四维点云数组。
        two_d_points: (M, 2) 二维点云数组，用于辅助定义对称轴。
        line_point: (2,) 二维直线上的一个点。
        line_vector: (2,) 二维直线的方向向量。

    返回：
        all_points: (2N, 4) 原点云与镜像点云拼接后的结果。
    """
    points = np.asarray(points)
    coords = points[:, :3]
    attrs = points[:, 3:]

    # 按 z 降序排列
    sorted_indices = np.argsort(coords[:, 2])[::-1]
    sorted_coords = coords[sorted_indices]
    sorted_attrs = attrs[sorted_indices]

    mirrored_points = []

    for i in range(len(sorted_coords)):
        x, y, z = sorted_coords[i]
        attr = sorted_attrs[i][0]

        # 镜像 xy 坐标
        mirrored_x, mirrored_y = mirror_point_over_line([x, y], line_point, line_vector)
        mirrored_points.append([mirrored_x, mirrored_y, z, attr])

    mirrored_points = np.array(mirrored_points)

    # 拼接原始点云和镜像点云
    all_points = np.vstack((points, mirrored_points))

    return all_points


def mirror_point_over_line(point, line_point, line_vector):
    """
    将一个二维点关于指定的二维直线进行镜像变换。

    参数：
        point: (2,) 二维点坐标。
        line_point: (2,) 直线上的一点。
        line_vector: (2,) 直线方向向量。

    返回：
        mirrored_point: (2,) 镜像变换后的点坐标。
    """
    point = np.asarray(point)
    line_point = np.asarray(line_point)
    line_vector = np.asarray(line_vector)

    # 单位化方向向量
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)

    # 点到直线方向的向量
    vector_to_line = point - line_point

    # 投影长度
    proj_len = np.dot(vector_to_line, line_vector_normalized)

    # 投影点在直线上的位置
    proj_point = line_point + proj_len * line_vector_normalized

    # 镜像点 = 投影点 + (投影点 - 原点)
    mirrored_point = 2 * proj_point - point

    return mirrored_point

def cloud_point_completion(points, colors):
    projected_pcd = project_to_xy(points)
    _, center = compute_2d_min_obb(projected_pcd)

    mirrored = mirror_point_cloud_over_line(points, center, )

if __name__ == "__main__":

    # 示例四维点云 (x, y, z, 强度)
    points = np.array([
        [1.0, 2.0, 0.5, 0.1],
        [2.0, 3.0, 0.5, 0.2],
        [3.0, 4.0, 1.5, 0.3],
        [1.0, 5.0, 1.0, 0.1]
    ])

    projected = project_to_xy(points)
    print(projected)
    _, center = compute_2d_min_obb(projected)

    line_point = [0.0, 0.0]
    line_vector = [0, 1.0]

    mirrored = mirror_point_cloud_over_line(points, line_point, line_vector)
    print(mirrored)
