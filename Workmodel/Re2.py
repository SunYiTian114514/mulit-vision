import copy
import numpy as np
import open3d as o3d


def fit_plane_ransac(pcd, distance_threshold=0.005, ransac_n=3, num_iterations=3000):
    """
    返回:
        plane_model: [a, b, c, d]
        inlier_cloud: 平面内点
        outlier_cloud: 非平面点
        inlier_idx: 内点索引
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud, inliers


def get_plane_height_from_inliers(inlier_cloud, use_median=True):
    """
    假设桌面已经大致水平，取平面内点的 z 中位数/均值作为桌面高度
    """
    pts = np.asarray(inlier_cloud.points)
    if len(pts) == 0:
        raise ValueError("平面内点为空，无法计算高度")
    z = pts[:, 2]
    return np.median(z) if use_median else np.mean(z)


def align_parallel_planes_to_same_z(pcd_list, ref_mode="first", distance_threshold=0.005):
    """
    只做“平行桌面对齐到同一个平面”
    前提：这些桌面已经互相平行，最好已经基本水平

    参数:
        pcd_list: [pcd1, pcd2, pcd3, ...]
        ref_mode:
            - "first": 以第一个点云桌面为参考
            - "zero": 统一平移到 z=0
        distance_threshold: RANSAC 拟合平面阈值

    返回:
        aligned_pcds: 对齐后的点云列表
        heights: 每个点云原始桌面高度
        dz_list: 每个点云施加的 z 平移
    """
    aligned_pcds = []
    heights = []
    plane_models = []

    # 1) 拟合每个桌面，取桌面高度
    for i, pcd in enumerate(pcd_list):
        plane_model, inlier_cloud, _, _ = fit_plane_ransac(
            pcd,
            distance_threshold=distance_threshold
        )
        h = get_plane_height_from_inliers(inlier_cloud, use_median=True)
        plane_models.append(plane_model)
        heights.append(h)
        print(f"[Cloud {i}] plane = {plane_model}, height(z) = {h:.6f}")

    # 2) 选择目标平面高度
    if ref_mode == "first":
        z_ref = heights[0]
    elif ref_mode == "zero":
        z_ref = 0.0
    else:
        raise ValueError("ref_mode 只能是 'first' 或 'zero'")

    # 3) 只沿 z 方向平移
    dz_list = []
    for i, pcd in enumerate(pcd_list):
        dz = z_ref - heights[i]
        dz_list.append(dz)

        pcd_new = copy.deepcopy(pcd)
        pcd_new.translate((0.0, 0.0, dz))
        aligned_pcds.append(pcd_new)

        print(f"[Cloud {i}] apply dz = {dz:.6f}")

    return aligned_pcds, heights, dz_list


def visualize_with_frame(pcd_list, window_name="Aligned Parallel Planes"):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([axis] + pcd_list, window_name=window_name)


if __name__ == "__main__":
    # 换成你的文件
    ply1 = r"outputs_align_horizontal/pcd1_rotated_horizontal.ply"
    ply2 = r"outputs_align_horizontal/pcd2_rotated_horizontal.ply"
    ply3 = r"outputs_align_horizontal/pcd3_rotated_horizontal.ply"

    pcd1 = o3d.io.read_point_cloud(ply1)
    pcd2 = o3d.io.read_point_cloud(ply2)
    pcd3 = o3d.io.read_point_cloud(ply3)

    aligned_pcds, heights, dz_list = align_parallel_planes_to_same_z(
        [pcd1, pcd2, pcd3],
        ref_mode="first",      # 或 "zero"
        distance_threshold=0.005
    )

    # 保存
    o3d.io.write_point_cloud("cam1_same_plane.ply", aligned_pcds[0])
    o3d.io.write_point_cloud("cam2_same_plane.ply", aligned_pcds[1])
    o3d.io.write_point_cloud("cam3_same_plane.ply", aligned_pcds[2])

    visualize_with_frame(aligned_pcds, "Three Parallel Planes -> Same Plane")