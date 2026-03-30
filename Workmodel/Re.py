import copy
import os
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================================================
# 1) 这里改成你的三张点云
# =========================================================

PLY1 = r"outputs_world_color/730.ply"
PLY2 = r"outputs_world_color/919.ply"
PLY3 = r"outputs_world_color/933.ply"

OUTPUT_DIR = "outputs_align_horizontal"

# 是否保存旋转后的点云
SAVE_ROTATED_PCD = True

# 是否显示可视化
SHOW_VIS = True

# 可视化设置
POINT_SIZE = 2.0
BG = "white"   # "black" or "white"
SHOW_WORLD_AXIS = True
WORLD_AXIS_SIZE = 0.05

# 平面拟合参数
PLANE_DISTANCE_THRESHOLD = 0.005   # 单位若是米，5mm 先试
PLANE_RANSAC_N = 3
PLANE_NUM_ITER = 300000

# 可选预处理
VOXEL_SIZE = 0.0        # 比如 0.002；0 表示不下采样
REMOVE_OUTLIER = False
NB_NEIGHBORS = 20
STD_RATIO = 2.0

# 目标水平法向
TARGET_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# =========================================================
# 2) 下面一般不用改
# =========================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")
    return pcd


def preprocess_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.0,
    remove_outlier: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    out = copy.deepcopy(pcd)

    if voxel > 0:
        print(f"[INFO] voxel_down_sample: voxel_size={voxel}")
        out = out.voxel_down_sample(voxel_size=voxel)

    if remove_outlier:
        print(f"[INFO] remove_statistical_outlier: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        out, _ = out.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

    return out


def print_pcd_info(name: str, pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ct = pts.mean(axis=0)
    extent = mx - mn
    diag = float(np.linalg.norm(extent))

    print(f"[INFO] {name}:")
    print(f"       points = {len(pts)}")
    print(f"       min    = {mn}")
    print(f"       max    = {mx}")
    print(f"       center = {ct}")
    print(f"       extent = {extent}")
    print(f"       diag   = {diag:.6f}")


def normalize_plane(plane_model):
    """
    平面方程 ax + by + cz + d = 0
    归一化使法向长度为 1
    """
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Plane normal norm too small.")
    return np.array([a / norm, b / norm, c / norm, d / norm], dtype=np.float64)


def fit_largest_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold=0.005,
    ransac_n=3,
    num_iterations=3000
):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    plane = normalize_plane(plane_model)
    plane_pcd = pcd.select_by_index(inliers)
    remain_pcd = pcd.select_by_index(inliers, invert=True)
    return plane, plane_pcd, remain_pcd, inliers


def align_normal_sign(plane, ref):
    """
    让法向尽量朝 ref 同向
    """
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    if np.dot(n, ref) < 0:
        return np.array([-a, -b, -c, -d], dtype=np.float64)
    return plane


def skew(v):
    """
    反对称矩阵 [v]_x
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)


def rotation_from_a_to_b(a, b):
    """
    计算将单位向量 a 旋到单位向量 b 的旋转矩阵
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # 几乎同向
    if s < 1e-12 and c > 0:
        return np.eye(3, dtype=np.float64)

    # 几乎反向：选一个垂直轴转 180°
    if s < 1e-12 and c < 0:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        vx = skew(v)
        return np.eye(3) + 2.0 * (vx @ vx)

    vx = skew(v)
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
    return R


def make_T_from_R(R):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def transform_pcd(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    out = copy.deepcopy(pcd)
    out.transform(T)
    return out


def colorize_plane(plane_pcd, color):
    out = copy.deepcopy(plane_pcd)
    out.paint_uniform_color(color)
    return out


def create_world_axis(size: float):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])


def fit_view(vis: o3d.visualization.Visualizer, pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent))
    if diag < 1e-8:
        diag = 1.0

    vc = vis.get_view_control()
    vc.set_lookat(center.tolist())
    vc.set_front([0.0, 0.0, -1.0])
    vc.set_up([0.0, -1.0, 0.0])

    if diag < 0.2:
        zoom = 0.8
    elif diag < 1.0:
        zoom = 0.6
    elif diag < 5.0:
        zoom = 0.45
    else:
        zoom = 0.3
    vc.set_zoom(zoom)


def visualize(geometries, merged_for_fit, title="Plane Alignment"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1500, height=950)

    for g in geometries:
        vis.add_geometry(g)

    if SHOW_WORLD_AXIS:
        vis.add_geometry(create_world_axis(WORLD_AXIS_SIZE))

    opt = vis.get_render_option()
    opt.point_size = float(POINT_SIZE)
    opt.light_on = True

    if BG == "white":
        opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    vis.poll_events()
    vis.update_renderer()
    fit_view(vis, merged_for_fit)

    print("\n[INFO] Controls:")
    print("  Left mouse   : rotate")
    print("  Wheel        : zoom")
    print("  Right mouse  : translate")
    print("  Ctrl + Left  : roll")
    print("  Q / ESC      : quit")

    vis.run()
    vis.destroy_window()


def process_one(name, ply_path, plane_color, target_normal):
    pcd = load_pcd(ply_path)
    pcd = preprocess_pcd(
        pcd,
        voxel=VOXEL_SIZE,
        remove_outlier=REMOVE_OUTLIER,
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )

    print_pcd_info(name, pcd)

    plane, plane_pcd, remain_pcd, inliers = fit_largest_plane(
        pcd,
        distance_threshold=PLANE_DISTANCE_THRESHOLD,
        ransac_n=PLANE_RANSAC_N,
        num_iterations=PLANE_NUM_ITER
    )

    # 统一法向朝向
    plane = align_normal_sign(plane, target_normal)
    normal = plane[:3]

    # 计算把当前法向转到水平面的旋转矩阵
    R_align = rotation_from_a_to_b(normal, target_normal)
    T_align = make_T_from_R(R_align)

    pcd_rot = transform_pcd(pcd, T_align)
    plane_pcd_rot = transform_pcd(plane_pcd, T_align)

    plane_vis = colorize_plane(plane_pcd, plane_color)
    plane_vis_rot = colorize_plane(plane_pcd_rot, plane_color)

    return {
        "name": name,
        "pcd": pcd,
        "plane": plane,
        "normal": normal,
        "plane_pcd": plane_pcd,
        "plane_vis": plane_vis,
        "R_align": R_align,
        "T_align": T_align,
        "pcd_rot": pcd_rot,
        "plane_vis_rot": plane_vis_rot,
        "inlier_count": len(inliers),
    }


def main():
    ensure_dir(OUTPUT_DIR)

    cam1 = process_one("pcd1", PLY1, [1.0, 0.2, 0.2], TARGET_NORMAL)
    cam2 = process_one("pcd2", PLY2, [0.2, 1.0, 0.2], TARGET_NORMAL)
    cam3 = process_one("pcd3", PLY3, [0.2, 0.4, 1.0], TARGET_NORMAL)

    cams = [cam1, cam2, cam3]

    print("\n==================== PLANE RESULTS ====================")
    for cam in cams:
        print(f"[PLANE] {cam['name']}")
        print(f"        plane  = {cam['plane']}")
        print(f"        normal = {cam['normal']}")
        print(f"        inliers= {cam['inlier_count']}")
        print(f"        R_align =")
        print(cam["R_align"])
        print()

    # 保存旋转矩阵
    report_path = os.path.join(OUTPUT_DIR, "alignment_rotations.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for cam in cams:
            f.write(f"{cam['name']}\n")
            f.write(f"plane   = {cam['plane']}\n")
            f.write(f"normal  = {cam['normal']}\n")
            f.write(f"inliers = {cam['inlier_count']}\n")
            f.write("R_align =\n")
            f.write(np.array2string(cam["R_align"], precision=8, suppress_small=True))
            f.write("\n\n")

    print(f"[INFO] Saved rotation report: {report_path}")

    if SAVE_ROTATED_PCD:
        for cam in cams:
            out_path = os.path.join(OUTPUT_DIR, f"{cam['name']}_rotated_horizontal.ply")
            o3d.io.write_point_cloud(out_path, cam["pcd_rot"], write_ascii=True)
            print(f"[INFO] Saved: {out_path}")

    if SHOW_VIS:
        # 原始
        merged_raw = o3d.geometry.PointCloud()
        merged_raw += cam1["pcd"]
        merged_raw += cam2["pcd"]
        merged_raw += cam3["pcd"]

        visualize(
            geometries=[
                cam1["pcd"], cam2["pcd"], cam3["pcd"],
                cam1["plane_vis"], cam2["plane_vis"], cam3["plane_vis"]
            ],
            merged_for_fit=merged_raw,
            title="Original Point Clouds + Fitted Planes"
        )

        # 旋转后
        merged_rot = o3d.geometry.PointCloud()
        merged_rot += cam1["pcd_rot"]
        merged_rot += cam2["pcd_rot"]
        merged_rot += cam3["pcd_rot"]

        visualize(
            geometries=[
                cam1["pcd_rot"], cam2["pcd_rot"], cam3["pcd_rot"],
                cam1["plane_vis_rot"], cam2["plane_vis_rot"], cam3["plane_vis_rot"]
            ],
            merged_for_fit=merged_rot,
            title="Rotated Point Clouds Aligned To Horizontal"
        )


if __name__ == "__main__":
    main()