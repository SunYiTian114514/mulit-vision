import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DepthPro depth + RGB image to world-coordinate colored point cloud"
    )

    # 输入
    parser.add_argument("--depth_npy", type=str, required=True, help="Input depth map .npy, shape (H,W)")
    parser.add_argument("--rgb", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--mask_npy", type=str, default=None, help="Optional binary mask .npy, shape (H,W)")
    parser.add_argument("--output_dir", type=str, default="outputs_world_color", help="Output directory")

    # 内参
    parser.add_argument("--fx", type=float, required=True, help="Camera fx")
    parser.add_argument("--fy", type=float, required=True, help="Camera fy")
    parser.add_argument("--cx", type=float, required=True, help="Camera cx")
    parser.add_argument("--cy", type=float, required=True, help="Camera cy")

    # 深度范围
    parser.add_argument("--min_depth", type=float, default=0.0, help="Minimum valid depth")
    parser.add_argument("--max_depth", type=float, default=1e9, help="Maximum valid depth")

    # 外参模式
    parser.add_argument(
        "--extrinsic_mode",
        type=str,
        default="world_to_camera",
        choices=["world_to_camera", "camera_to_world"],
        help="Interpretation of input R/T"
    )

    # 坐标轴翻转
    parser.add_argument("--flip_y", action="store_true", help="Flip Y axis in camera coordinates before world transform")
    parser.add_argument("--flip_z", action="store_true", help="Flip Z axis in camera coordinates before world transform")

    # 输出控制
    parser.add_argument("--save_camera_ply", action="store_true", help="Also save camera-coordinate colored point cloud")
    parser.add_argument("--save_world_map_xyz_separate", action="store_true", help="Also save Xw/Yw/Zw as separate .npy files")

    return parser.parse_args()


# =========================
# 这里直接修改你的 R/T
# 默认按 world -> camera
# Pc = R @ Pw + T
# =========================

# R = np.array([
#     [0.5547, 0.8320, 3.3764e-04],
#     [-0.0066, 0.0040, 1],
#     [0.8320, -0.5547, 0.0077]
# ], dtype=np.float64)
#
# T = np.array([-16.2687, -0.0172, 260.7818], dtype=np.float64) / 1000

#
# 730的相机矩阵
# R = np.array([
#     [0.5547, 0.8320, 3.3764e-04],
#     [-0.0066, 0.0040, 1],
#     [0.8320, -0.5547, 0.0077]
# ], dtype=np.float64)
#
# T = np.array([-16.2687, -0.0172, 260.7818], dtype=np.float64) / 1000
# #
# 919的相机矩阵
# R = np.array([
#         [0.3875, -0.9218, 0.0040],
#         [0.6631, 0.2818, 0.6934],
#         [-0.6404, -0.2661, 0.7205]
#     ], dtype=np.float64)
# T = np.array([1.9990, -40.0983, 210.5670], dtype=np.float64) / 1000

# 933的相机矩阵
R = np.array([
    [-0.9988, 0.0474, -0.0084],
    [-0.0188, -0.2240, 0.9744],
    [0.0443, 0.9734, 0.2247]
], dtype=np.float64)

T = np.array([-2.6113, -4.5020, 264.5247], dtype=np.float64) / 1000





# =========================
# 下面一般不用改
# =========================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_depth(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be shape (H,W), got {depth.shape}")
    return depth.astype(np.float32)


def load_rgb(path: str, expected_shape_hw) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"RGB image not found: {path}")
    rgb = np.array(Image.open(path).convert("RGB"))
    if rgb.shape[:2] != expected_shape_hw:
        raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth shape {expected_shape_hw}")
    return rgb.astype(np.uint8)


def load_mask(path: str, expected_shape) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = np.load(path)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match depth shape {expected_shape}")
    return (mask > 0).astype(np.uint8)


def build_T_from_R_t(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3)

    if R.shape != (3, 3):
        raise ValueError(f"R must be (3,3), got {R.shape}")
    if T.shape != (3,):
        raise ValueError(f"T must be (3,), got {T.shape}")

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


def invert_world_to_camera(R_wc: np.ndarray, t_wc: np.ndarray):
    """
    已知:
        Pc = R_wc @ Pw + t_wc
    求:
        Pw = R_cw @ Pc + t_cw
    """
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc.reshape(3)
    return R_cw, t_cw


def make_flip_matrix(flip_y: bool, flip_z: bool) -> np.ndarray:
    F = np.eye(3, dtype=np.float64)
    if flip_y:
        F[1, 1] = -1.0
    if flip_z:
        F[2, 2] = -1.0
    return F


def compute_valid_mask(depth: np.ndarray, min_depth: float, max_depth: float, mask: np.ndarray | None):
    valid = np.isfinite(depth) & (depth > 0)
    valid &= (depth >= min_depth) & (depth <= max_depth)
    if mask is not None:
        valid &= (mask > 0)
    return valid


def depth_to_camera_points_and_map(depth, fx, fy, cx, cy, valid_mask, flip_y=False, flip_z=False):
    """
    返回：
        cam_map: (H,W,3)
        cam_points: (N,3)
    """
    h, w = depth.shape
    ys, xs = np.indices((h, w), dtype=np.float64)

    Z = depth.astype(np.float64)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    cam_map = np.stack([X, Y, Z], axis=-1)  # (H,W,3)

    F = make_flip_matrix(flip_y, flip_z)
    cam_map = cam_map @ F.T

    cam_points = cam_map[valid_mask]
    return cam_map, cam_points


def camera_to_world_map_and_points(cam_map, valid_mask, R_cw, t_cw):
    """
    cam_map: (H,W,3)
    valid_mask: (H,W)
    """
    h, w, _ = cam_map.shape
    cam_flat = cam_map.reshape(-1, 3).T  # (3,N)

    world_flat = R_cw @ cam_flat + t_cw.reshape(3, 1)
    world_map = world_flat.T.reshape(h, w, 3)

    world_points = world_map[valid_mask]
    return world_map, world_points


def save_ply_ascii_xyzrgb(points: np.ndarray, colors: np.ndarray, path: str):
    if len(points) != len(colors):
        raise ValueError("points and colors must have same length")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def print_stats(name: str, points: np.ndarray):
    if len(points) == 0:
        print(f"[INFO] {name}: empty")
        return
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    ct = points.mean(axis=0)
    print(f"[INFO] {name}:")
    print(f"       count  = {len(points)}")
    print(f"       min    = {mn}")
    print(f"       max    = {mx}")
    print(f"       center = {ct}")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("[INFO] Loading depth...")
    depth = load_depth(args.depth_npy)
    h, w = depth.shape
    print(f"[INFO] Depth shape: {depth.shape}")

    print("[INFO] Loading RGB...")
    rgb = load_rgb(args.rgb, (h, w))

    mask = None
    if args.mask_npy is not None:
        print("[INFO] Loading mask...")
        mask = load_mask(args.mask_npy, depth.shape)

    print("[INFO] Computing valid mask...")
    valid_mask = compute_valid_mask(
        depth=depth,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        mask=mask
    )
    print(f"[INFO] Valid pixels: {int(valid_mask.sum())}")

    if valid_mask.sum() == 0:
        raise RuntimeError("No valid pixels found after depth/mask filtering.")

    # 解释外参
    if args.extrinsic_mode == "world_to_camera":
        print("[INFO] Input extrinsic mode: world_to_camera")
        R_cw, t_cw = invert_world_to_camera(R, T)
    else:
        print("[INFO] Input extrinsic mode: camera_to_world")
        R_cw = np.asarray(R, dtype=np.float64)
        t_cw = np.asarray(T, dtype=np.float64).reshape(3)

    T_cw = build_T_from_R_t(R_cw, t_cw)

    print("[INFO] Camera-to-world transform:")
    print(np.array2string(T_cw, precision=6, suppress_small=True))

    print("[INFO] Back-projecting depth to camera coordinates...")
    cam_map, cam_points = depth_to_camera_points_and_map(
        depth=depth,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        valid_mask=valid_mask,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )

    print_stats("camera points", cam_points)

    print("[INFO] Transforming camera coordinates to world coordinates...")
    world_map, world_points = camera_to_world_map_and_points(
        cam_map=cam_map,
        valid_mask=valid_mask,
        R_cw=R_cw,
        t_cw=t_cw
    )

    print_stats("world points", world_points)

    # 提取颜色
    colors = rgb[valid_mask]  # (N,3), uint8

    # 保存 world_map
    world_map_path = os.path.join(args.output_dir, "world_map.npy")
    np.save(world_map_path, world_map.astype(np.float32))
    print(f"[INFO] Saved world_map: {world_map_path}")

    if args.save_world_map_xyz_separate:
        np.save(os.path.join(args.output_dir, "world_x.npy"), world_map[..., 0].astype(np.float32))
        np.save(os.path.join(args.output_dir, "world_y.npy"), world_map[..., 1].astype(np.float32))
        np.save(os.path.join(args.output_dir, "world_z.npy"), world_map[..., 2].astype(np.float32))
        print("[INFO] Saved world_x.npy, world_y.npy, world_z.npy")

    # 保存世界系彩色点云
    world_ply_path = os.path.join(args.output_dir, "world_points_color.ply")
    save_ply_ascii_xyzrgb(world_points, colors, world_ply_path)
    print(f"[INFO] Saved world-coordinate colored point cloud: {world_ply_path}")

    # 可选保存相机系彩色点云
    if args.save_camera_ply:
        cam_ply_path = os.path.join(args.output_dir, "camera_points_color.ply")
        save_ply_ascii_xyzrgb(cam_points, colors, cam_ply_path)
        print(f"[INFO] Saved camera-coordinate colored point cloud: {cam_ply_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()