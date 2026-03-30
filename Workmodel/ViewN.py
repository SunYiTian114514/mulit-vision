import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open3D viewer for world-coordinate point clouds"
    )
    parser.add_argument("--ply", type=str, required=True, help="Path to .ply file")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point size")
    parser.add_argument("--bg", type=str, default="black", choices=["black", "white"], help="Background color")

    parser.add_argument("--show_axis", action="store_true", help="Show coordinate frame")
    parser.add_argument("--axis_size", type=float, default=0.1, help="Coordinate frame size")

    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size, 0 to disable")
    parser.add_argument("--remove_outlier", action="store_true", help="Enable statistical outlier removal")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Neighbors for outlier removal")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Std ratio for outlier removal")

    parser.add_argument("--estimate_normals", action="store_true", help="Estimate normals")
    parser.add_argument("--normal_radius", type=float, default=0.02, help="Radius for normal estimation")
    parser.add_argument("--normal_max_nn", type=int, default=30, help="Max NN for normal estimation")
    parser.add_argument("--show_normals", action="store_true", help="Show normals")

    parser.add_argument("--center_origin", action="store_true", help="Translate point cloud center to origin")
    parser.add_argument("--save_screenshot", type=str, default=None, help="Optional screenshot path")

    return parser.parse_args()


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print(f"[ERROR] Loaded point cloud is empty: {path}")
        sys.exit(1)

    return pcd


def print_stats(name: str, pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)
    center = pts.mean(axis=0)
    extent = max_bound - min_bound
    diag = float(np.linalg.norm(extent))

    print(f"[INFO] {name}:")
    print(f"       points = {len(pts)}")
    print(f"       min    = {min_bound}")
    print(f"       max    = {max_bound}")
    print(f"       center = {center}")
    print(f"       extent = {extent}")
    print(f"       diag   = {diag:.6f}")

    if len(pcd.colors) > 0:
        print("[INFO] RGB colors detected.")
    else:
        print("[INFO] No RGB colors detected.")

    if pcd.has_normals():
        print("[INFO] Normals available.")
    else:
        print("[INFO] Normals not available.")


def preprocess_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.0,
    remove_outlier: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)

    if voxel > 0:
        print(f"[INFO] Voxel downsample: {voxel}")
        out = out.voxel_down_sample(voxel_size=voxel)
        print(f"[INFO] After voxel downsample: {len(np.asarray(out.points))} points")

    if remove_outlier:
        print(f"[INFO] Statistical outlier removal: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        out, _ = out.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"[INFO] After outlier removal: {len(np.asarray(out.points))} points")

    return out


def estimate_normals_if_needed(
    pcd: o3d.geometry.PointCloud,
    estimate_normals: bool,
    normal_radius: float,
    normal_max_nn: int
) -> o3d.geometry.PointCloud:
    if estimate_normals:
        print(f"[INFO] Estimating normals: radius={normal_radius}, max_nn={normal_max_nn}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn
            )
        )
        pcd.normalize_normals()
    return pcd


def center_to_origin(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)
    center = out.get_center()
    print(f"[INFO] Translating center to origin: {center}")
    out.translate(-center)
    return out


def create_axis(size: float):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])


def fit_camera_to_geometry(vis: o3d.visualization.Visualizer, pcd: o3d.geometry.PointCloud):
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


def main():
    args = parse_args()

    print(f"[INFO] Loading point cloud: {args.ply}")
    pcd = load_point_cloud(args.ply)
    print_stats("original", pcd)

    pcd = preprocess_pcd(
        pcd=pcd,
        voxel=args.voxel,
        remove_outlier=args.remove_outlier,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )

    if args.center_origin:
        pcd = center_to_origin(pcd)

    pcd = estimate_normals_if_needed(
        pcd=pcd,
        estimate_normals=args.estimate_normals,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
    )

    print_stats("processed", pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D World Point Viewer", width=1400, height=900)
    vis.add_geometry(pcd)

    if args.show_axis:
        axis = create_axis(args.axis_size)
        vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.light_on = True
    opt.point_show_normal = bool(args.show_normals)

    if args.bg == "white":
        opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    vis.poll_events()
    vis.update_renderer()
    fit_camera_to_geometry(vis, pcd)

    print("\n[INFO] Controls:")
    print("  Left mouse   : rotate")
    print("  Wheel        : zoom")
    print("  Right mouse  : translate")
    print("  Ctrl + Left  : roll")
    print("  R            : reset view")
    print("  Q / ESC      : quit")

    if args.save_screenshot:
        screenshot_path = Path(args.save_screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        vis.poll_events()
        vis.update_renderer()
        ok = vis.capture_screen_image(str(screenshot_path), do_render=True)
        print(f"[INFO] Screenshot saved: {screenshot_path} (success={ok})")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()