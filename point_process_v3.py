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

def plot_obb(points, box, center):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], label='Point Cloud')
    box_loop = np.vstack([box, box[0]])  # 闭合 OBB 线
    plt.plot(box_loop[:, 0], box_loop[:, 1], 'r-', label='OBB')
    plt.plot(center[0], center[1], 'go', label='Center')
    plt.axis('equal')
    plt.legend()
    plt.title("2D OBB and Center")
    plt.show()

def generate_cuboid_point_cloud(length, width, height, resolution=0.01):
    """
    生成一个长方体的表面点云，模拟点云相机拍摄效果。
    
    参数：
        length: 沿X轴的长度
        width:  沿Y轴的宽度
        height: 沿Z轴的高度
        resolution: 点的间隔（越小越密集）
    
    返回：
        (N, 3) 的点云 NumPy 数组
    """
    x_vals = np.arange(0, length + resolution, resolution)
    y_vals = np.arange(0, width + resolution, resolution)
    z_vals = np.arange(0, height + resolution, resolution)

    # 每个面生成点云：6个面
    faces = []

    # Front and Back (z=0 and z=height)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = np.zeros_like(X)
    Z1 = np.full_like(X, height)
    faces.append(np.stack([X, Y, Z0], axis=-1).reshape(-1, 3))  # front
    faces.append(np.stack([X, Y, Z1], axis=-1).reshape(-1, 3))  # back

    # Top and Bottom (y=0 and y=width)
    X, Z = np.meshgrid(x_vals, z_vals)
    Y0 = np.zeros_like(X)
    Y1 = np.full_like(X, width)
    faces.append(np.stack([X, Y0, Z], axis=-1).reshape(-1, 3))  # bottom
    faces.append(np.stack([X, Y1, Z], axis=-1).reshape(-1, 3))  # top

    # Left and Right (x=0 and x=length)
    Y, Z = np.meshgrid(y_vals, z_vals)
    X0 = np.zeros_like(Y)
    X1 = np.full_like(Y, length)
    faces.append(np.stack([X0, Y, Z], axis=-1).reshape(-1, 3))  # left
    faces.append(np.stack([X1, Y, Z], axis=-1).reshape(-1, 3))  # right

    # 合并所有面点
    point_cloud = np.concatenate(faces, axis=0)

    return point_cloud


def project_point_cloud_to_plane(points, plane_normal, plane_point):
    """
    将点云投影到指定平面
    points: (N, 3) 的点云数组
    plane_normal: 平面法向量 (A, B, C)
    plane_point: 平面上一点 (x, y, z)
    """
    normal = np.array(plane_normal)
    normal = normal / np.linalg.norm(normal)  # 单位化
    plane_point = np.array(plane_point)

    # 向量从平面点指向点云中每个点
    vecs = points - plane_point
    # 投影长度（点到平面的距离）
    distances = np.dot(vecs, normal)
    # 投影点 = 原始点 - 距离 * 法向量
    projected_points = points - np.outer(distances, normal)

    return projected_points

def project_to_xy(points):
    """
    将三维点云投影到 XY 平面（z=0）
    points: (N, 3) 的 NumPy 数组
    返回: (N, 2) 的二维点（x, y）
    """
    return points[:, :2]  # 取 x 和 y 分量

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# 显示点云

def generate_visible_cuboid_point_cloud(length, width, height, resolution=0.01):
    """
    只生成长方体的三个可见面：前面 (z=0)，顶面 (y=width)，右面 (x=length) 的点云。
    自动去除边缘重叠点，避免多面重复采样。
    
    参数：
        length: 沿 X 轴的长度
        width:  沿 Y 轴的宽度
        height: 沿 Z 轴的高度
        resolution: 点间隔
        
    返回：
        (N, 3) 点云数组
    """
    # 通用轴范围
    x_vals = np.arange(0, length + resolution, resolution)
    y_vals = np.arange(0, width + resolution, resolution)
    z_vals = np.arange(0, height + resolution, resolution)

    point_set = []

    # -------- 前面 (z=0) --------
    Xf, Yf = np.meshgrid(x_vals, y_vals)
    Zf = np.zeros_like(Xf)
    front_face = np.stack([Xf, Yf, Zf], axis=-1).reshape(-1, 3)
    point_set.append(front_face)

    # -------- 顶面 (y=width), 去除已在 front 的点 (z=0 部分) --------
    Xt, Zt = np.meshgrid(x_vals, z_vals)
    Yt = np.full_like(Xt, width)
    top_face = np.stack([Xt, Yt, Zt], axis=-1).reshape(-1, 3)
    top_face = top_face[top_face[:, 2] > 0]  # 仅保留 z>0，避免与 front 重叠
    point_set.append(top_face)

    # -------- 右面 (x=length), 去除已在 front 或 top 的点 --------
    Yr, Zr = np.meshgrid(y_vals, z_vals)
    Xr = np.full_like(Yr, length)
    right_face = np.stack([Xr, Yr, Zr], axis=-1).reshape(-1, 3)
    right_face = right_face[(right_face[:, 2] > 0) & (right_face[:, 1] < width)]  # z>0 且 y<width
    point_set.append(right_face)

    # 合并
    point_cloud = np.concatenate(point_set, axis=0)
    return point_cloud

def generate_cylinder_point_cloud(radius, height, resolution=0.01):
    """
    生成圆柱体的表面点云，包括顶部、底部和侧面。

    参数：
        radius: 圆柱的半径
        height: 圆柱的高
        resolution: 点之间的间隔距离

    返回：
        (N, 3) 点云数组
    """
    point_set = []

    # 生成顶部面（z=height）
    theta = np.linspace(0, 2*np.pi, int(2*np.pi*radius/resolution))
    r = np.linspace(0, radius, int(radius/resolution))
    theta_grid, r_grid = np.meshgrid(theta, r)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    z = np.full_like(x, height)
    top_face = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    point_set.append(top_face)

    # 生成底部面（z=0）
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    z = np.zeros_like(x)
    bottom_face = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    point_set.append(bottom_face)

    # 生成侧面（圆柱侧面）
    theta = np.linspace(0, 2*np.pi, int(2*np.pi*radius/resolution))
    z_vals = np.linspace(0, height, int(height/resolution))
    theta_grid, z_grid = np.meshgrid(theta, z_vals)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = z_grid
    side_face = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    point_set.append(side_face)

    # 合并所有点云
    point_cloud = np.concatenate(point_set, axis=0)

    return point_cloud

def combine_cuboid_and_cylinder_point_cloud(cuboid_length, cuboid_width, cuboid_height, cylinder_radius, cylinder_height, cylinder_center, resolution=0.01):
    # 生成长方体点云
    cuboid_pc = generate_visible_cuboid_point_cloud(cuboid_length, cuboid_width, cuboid_height, resolution)
    
    # 生成圆柱点云
    cylinder_pc = generate_cylinder_point_cloud(cylinder_radius, cylinder_height, resolution)
    # 将圆柱的底部与长方体顶部对齐，并将圆柱底部的圆心加在指定位置
    cylinder_pc[:, 2] += cuboid_height  # 将圆柱的底部提升到长方体的顶部
    cylinder_pc[:, 0] += cylinder_center[0]  # 将圆柱的圆心在x轴上移动到指定位置
    cylinder_pc[:, 1] += cylinder_center[1]  # 将圆柱的圆心在y轴上移动到指定位置
    
    # 合并两个点云
    combined_pc = np.concatenate([cuboid_pc, cylinder_pc], axis=0)
    
    return combined_pc


def stack_2d_onto_3d(points_2d, points_3d, center_2d, center_3d):
    """
    将二维点云按三维点云的实际Z层进行堆叠。
    
    参数：
        points_2d: (N, 2) 二维点云
        points_3d: (M, 3) 三维点云
        center_2d: (2,) 二维点云几何中心
        center_3d: (3,) 三维点云几何中心

    返回：
        combined_points: 合并后的三维点云 (原始点云 + 所有堆叠层)
    """
    # 提取唯一的 Z 层，并从高到低排序
    unique_z_levels = np.unique(points_3d[:, 2])
    unique_z_levels = np.sort(unique_z_levels)[::-1]

    # 将2D点云居中，并变为3D（z轴为0）
    points_2d_centered = points_2d - center_2d
    points_2d_3d = np.hstack([points_2d_centered, np.zeros((points_2d.shape[0], 1))])

    # 在每一个实际存在的Z层堆叠二维点云
    stacked_layers = []
    for z in unique_z_levels:
        layer = points_2d_3d.copy()
        layer[:, 2] = z
        # 平移至3D中心位置
        layer[:, 0] += center_3d[0]
        layer[:, 1] += center_3d[1]
        stacked_layers.append(layer)

    stacked_cloud = np.vstack(stacked_layers)

    # 合并原始点云
    combined = np.vstack([points_3d, stacked_cloud])
    return combined


def mirror_point_cloud_over_plane(points, plane_normal, plane_point):
    """
    将三维点云关于指定平面进行镜像变换。

    参数：
        points: (N, 3) 三维点云数组。
        plane_normal: (3,) 平面的法向量。
        plane_point: (3,) 平面上的一个点。

    返回：
        mirrored_points: (N, 3) 镜像变换后的三维点云。
    """
    # 将输入转换为numpy数组
    points = np.asarray(points)
    plane_normal = np.asarray(plane_normal)
    plane_point = np.asarray(plane_point)
    
    # 如果平面点只有两个坐标，添加第三个坐标（默认为0）
    if plane_point.shape[0] == 2:
        plane_point = np.append(plane_point, 0.0)
    
    # 确保平面法向量是单位向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # 计算每个点到平面的距离
    vector_to_plane = points - plane_point
    distance = np.dot(vector_to_plane, plane_normal)
    
    # 计算镜像点
    mirrored_points = points - 2 * distance[:, np.newaxis] * plane_normal
    
    return mirrored_points

def get_half_height(points):
    """
    输入: points (N, 3) 的三维点云
    返回: 点云高度的一半（z轴最大值-最小值的一半）
    """
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    height = z_max - z_min
    return height / 2


def mirror_point_cloud_over_line(points, two_d_points, line_point, line_vector):
    """
    将三维点云按照z轴从高到低逐层分解，并将每层的xy坐标关于指定的二维直线对称轴进行镜像操作，最后将镜像后的层重新叠加在一起。

    参数：
        points: (N, 3) 三维点云数组。
        two_d_points: (M, 2) 二维点云数组，用于确定对称轴的位置。
        line_point: (2,) 二维直线上的一个点。
        line_vector: (2,) 二维直线的方向向量。

    返回：
        mirrored_points: (N, 3) 镜像变换后的三维点云。
    """
    # 对三维点云的z坐标进行排序，从高到低
    sorted_indices = np.argsort(points[:, 2])[::-1]
    sorted_points = points[sorted_indices]

    # 计算每个点的xy坐标关于二维直线的镜像
    mirrored_points = []
    for point in sorted_points:
        x, y, z = point

        # 计算当前点的xy坐标关于二维直线的镜像
        mirrored_x, mirrored_y = mirror_point_over_line([x, y], line_point, line_vector)

        mirrored_points.append([mirrored_x, mirrored_y, z])

    mirrored_points = np.array(mirrored_points)

    return mirrored_points

def mirror_point_over_line(point, line_point, line_vector):
    """
    将一个二维点关于指定的二维直线进行镜像变换。

    参数：
        point: (2,) 二维点的坐标。
        line_point: (2,) 二维直线上的一个点。
        line_vector: (2,) 二维直线的方向向量。

    返回：
        mirrored_point: (2,) 镜像变换后的二维点坐标。
    """
    # 将输入转换为numpy数组
    point = np.asarray(point)
    line_point = np.asarray(line_point)
    line_vector = np.asarray(line_vector)

    # 计算直线的单位方向向量
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)

    # 计算点到直线的向量
    vector_to_line = point - line_point

    # 计算点到直线的距离
    distance = np.dot(vector_to_line, line_vector_normalized)

    # 计算镜像点
    mirrored_point = point - 2 * distance * line_vector_normalized

    return mirrored_point


# def intersection_of_pointclouds(points_a, points_b, center):
#     """
#     快速计算两个二维点云的凸包交集区域内的点。
#     使用 matplotlib.path.Path 加速点在多边形中的判断。
#     """
#     # 1. 生成凸包并求交集（仍使用 shapely）
#     poly_a = MultiPoint(points_a).convex_hull
#     poly_b = MultiPoint(points_b).convex_hull
#     intersection_poly = poly_a.intersection(poly_b)

#     # 2. 如果交集是空的，返回空
#     if intersection_poly.is_empty or not intersection_poly.geom_type == 'Polygon':
#         return np.empty((0, 2)), intersection_poly

#     # 3. 使用 matplotlib.path.Path 判断点是否在多边形内
#     poly_path = Path(np.array(intersection_poly.exterior.coords))
#     all_points = np.vstack((points_a, points_b))

#     inside_mask = poly_path.contains_points(all_points)
#     intersection_points = all_points[inside_mask]

#     return intersection_points, intersection_poly


# def intersection_of_pointclouds(points_a, points_b):
#     """
#     快速计算两个二维点云的凸包交集区域内的点。
#     使用 matplotlib.path.Path 加速点在多边形中的判断。
#     """

#     # 1. 计算凸包并求交集（shapely 非常快）
#     poly_a = MultiPoint(points_a).convex_hull
#     poly_b = MultiPoint(points_b).convex_hull
#     intersection_poly = poly_a.intersection(poly_b)

#     # 2. 空交集：提前返回
#     if intersection_poly.is_empty or intersection_poly.geom_type != 'Polygon':
#         return np.empty((0, 2)), intersection_poly

#     # 3. 如果交集是其中一个凸包：直接返回对应点集
#     if intersection_poly.equals(poly_a):
#         return points_a, intersection_poly
#     if intersection_poly.equals(poly_b):
#         return points_b, intersection_poly

#     # 4. 否则：判断哪些点在交集区域内（使用 Path 比 contains 快非常多）
#     all_points = np.concatenate((points_a, points_b), axis=0)
#     poly_coords = np.asarray(intersection_poly.exterior.coords)
#     poly_path = Path(poly_coords)
#     mask = poly_path.contains_points(all_points)
#     intersection_points = all_points[mask]

#     return intersection_points, intersection_poly

# def layer_intersection_of_point_clouds(points1, points2):
#     """
#     将两个三维点云按z轴从高到低逐层分解，并计算每层的二维点云交集，最后将交集重新叠加形成新的三维点云。

#     参数：
#         points1: (N, 3) 第一个三维点云数组。
#         points2: (M, 3) 第二个三维点云数组。

#     返回：
#         intersection_points: (K, 3) 交集点云数组。
#     """
    
#     # 利用np.unique返回索引，避免后续多次布尔索引
#     z_values1, idx1 = np.unique(points1[:, 2], return_inverse=True)
#     z_values2, idx2 = np.unique(points2[:, 2], return_inverse=True)
#     all_z_values = np.union1d(z_values1, z_values2)  # 已排序

#     # 预分组，减少每层的筛选时间

#     layer_dict1 = defaultdict(list)
#     layer_dict2 = defaultdict(list)
#     for i in range(len(points1)):
#         z = points1[i, 2]
#         layer_dict1[z].append(points1[i, :2])  # 只存储xy坐标
#     for i in range(len(points2)):
#         z = points2[i, 2]
#         layer_dict2[z].append(points2[i, :2])  # 只存储xy坐标

#     intersection_points = []

#     for z in all_z_values:
#         layer1 = np.array(layer_dict1.get(z, []))
#         layer2 = np.array(layer_dict2.get(z, []))

#         if len(layer1) == 0 or len(layer2) == 0:
#             continue  # 跳过没有点的层

#         # 计算当前层的交集
#         inter_points, _ = intersection_of_pointclouds(layer1, layer2)

#         # 将交集点添加到结果中，保持z坐标不变
#         if len(inter_points) > 0:
#             intersection_points.append(np.hstack((inter_points, np.full((len(inter_points), 1), z))))

#     # 合并所有层的交集点
#     if intersection_points:
#         intersection_points = np.vstack(intersection_points)
#     else:
#         intersection_points = np.empty((0, 3))  # 如果没有交集，返回空数组

#     return intersection_points


def rasterize_points(points, grid_size=512, bounds=None):
    """
    将二维点投影到二值图像掩膜上。
    """
    if bounds is None:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
    else:
        min_x, min_y, max_x, max_y = bounds

    # 构建像素索引
    scale_x = (grid_size - 1) / (max_x - min_x + 1e-8)
    scale_y = (grid_size - 1) / (max_y - min_y + 1e-8)
    x_idx = ((points[:, 0] - min_x) * scale_x).astype(int)
    y_idx = ((points[:, 1] - min_y) * scale_y).astype(int)

    # 生成掩膜图
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[y_idx, x_idx] = True

    return mask, (min_x, min_y, max_x, max_y), (scale_x, scale_y)

def points_in_mask(points, mask, bounds, scale):
    """
    判断哪些点在掩膜图区域中。
    """
    min_x, min_y, _, _ = bounds
    scale_x, scale_y = scale
    x_idx = ((points[:, 0] - min_x) * scale_x).astype(int)
    y_idx = ((points[:, 1] - min_y) * scale_y).astype(int)
    
    valid = (x_idx >= 0) & (x_idx < mask.shape[1]) & (y_idx >= 0) & (y_idx < mask.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[y_idx[valid], x_idx[valid]]
    return result

def rasterized_pointcloud_intersection(points1, points2, grid_size=512):
    """
    使用栅格法加速三维点云交集（逐层）。
    """
    result = []
    z_values = np.intersect1d(np.unique(points1[:, 2]), np.unique(points2[:, 2]))

    for z in z_values:
        layer1 = points1[points1[:, 2] == z][:, :2]
        layer2 = points2[points2[:, 2] == z][:, :2]
        if len(layer1) == 0 or len(layer2) == 0:
            continue

        # 使用统一 bounds 保证两图对齐
        all_points = np.vstack((layer1, layer2))
        mask1, bounds, scale = rasterize_points(layer1, grid_size)
        mask2, _, _ = rasterize_points(layer2, grid_size, bounds)
        mask_intersection = mask1 & mask2

        # 判断点是否在交集区域
        combined = np.vstack((layer1, layer2))
        in_mask = points_in_mask(combined, mask_intersection, bounds, scale)
        intersected = combined[in_mask]

        if intersected.size > 0:
            z_column = np.full((intersected.shape[0], 1), z)
            result.append(np.hstack((intersected, z_column)))

    if result:
        return np.vstack(result)
    else:
        return np.empty((0, 3))




if __name__ == "__main__":

    cylinder_center = (0.25, 0.15, 1.51)
    p = combine_cuboid_and_cylinder_point_cloud(0.5, 0.3, 1, 0.5, 0.5, cylinder_center, resolution=0.01)
    # p1 = generate_visible_cuboid_point_cloud(length=1.0, width=0.5, height=0.3, resolution=0.02) #长方体点云
    # print(p)

    projected = project_to_xy(p)
    # print(projected) # project是二维的

    # visualize_point_cloud(p)

    box, center = compute_2d_min_obb(projected)
    print("center:\n", center)
    half_height = get_half_height(p)
    # print("OBB四个顶点:\n", box)


    # plot_obb(projected, box, center)
    '''----------------------垂直生长---------------------'''
    # 将二维点云逐层叠加到三维点云上
    combined_points = stack_2d_onto_3d(projected, p, center, cylinder_center)
    # visualize_point_cloud(combined_points)


    '''----------------------水平生长---------------------'''
    # plane_normal = np.array([0, 0, 1])  # 平面法向量
    plane_normal1 = np.array([0, 0, 1])  # 平面法向量
    # plane_point = np.array([0.5, 0.25, 0.15])  # 平面上的一点
    plane_point = np.array([center[0], center[1], half_height])  # 组合成三维点]) 
    mirrored_points1 = mirror_point_cloud_over_plane(p, plane_normal1, plane_point)

    # visualize_point_cloud(mirrored_points)


    line_point = center
    line_vector1 = np.array([1.0, 0.0])  
    line_vector2 = np.array([0.0, 1.0])  # 直线的方向向量

    # 计算镜像点云
    mirrored_points1 = mirror_point_cloud_over_line(p, p[:, :2], line_point, line_vector1)
    mirrored_points2 = mirror_point_cloud_over_line(p, p[:, :2], line_point, line_vector2)


    union1 = np.vstack([p, mirrored_points1])
    # # 如果需要去重（每一行唯一），加上 axis=0
    union_unique1 = np.unique(union1, axis=0)

    # plane_normal2 = np.array([0, 1, 0])  # 平面法向量
    # mirrored_points2 = mirror_point_cloud_over_plane(p, plane_normal2, plane_point)
    # visualize_point_cloud(mirrored_points2)
    union2 = np.vstack([union_unique1, mirrored_points2])
    union_unique2 = np.unique(union2, axis=0)
    # visualize_point_cloud(union_unique2)
    # visualize_point_cloud(combined_points)

    # visualize_point_cloud(p)
    # visualize_point_cloud(mirrored_points1)
    # visualize_point_cloud(mirrored_points2)

    '''----------------------求取并集测试---------------------'''
    # projected1 = project_to_xy(p1)
    # inter_points, inter_poly = intersection_of_pointclouds(projected, projected1)

    # plt.figure(figsize=(6, 6))
    # plt.scatter(inter_points[:, 0], inter_points[:, 1], s=2, c='red', label='Merged Points')
    # plt.axis('equal')
    # plt.title('Merged 2D Point Cloud')
    # plt.legend()
    # plt.show()

    '''----------------------合成---------------------'''
    # 计算两个三维点云的逐层交集
    # intersection_points = layer_intersection_of_point_clouds(combined_points, union_unique2)
    start_time = time.time()
    intersection_points = rasterized_pointcloud_intersection(combined_points, union_unique2)
    end_time = time.time()
    print(f"Intersection calculation time: {end_time - start_time:.4f} seconds")
    visualize_point_cloud(intersection_points)