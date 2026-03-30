import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-precision inference script for DepthPro (Transformers version)"
    )
    parser.add_argument("--image", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    parser.add_argument("--model_name", type=str, default="apple/DepthPro-hf", help="HF model name")
    parser.add_argument("--use_fast_processor", action="store_true", help="Try fast image processor if available")

    # 深度范围
    parser.add_argument("--min_depth", type=float, default=0.05, help="Minimum valid depth in meters")
    parser.add_argument("--max_depth", type=float, default=20.0, help="Maximum valid depth in meters")

    # 后处理
    parser.add_argument("--bilateral", action="store_true", help="Apply bilateral filter to depth")
    parser.add_argument("--bilateral_d", type=int, default=7, help="Bilateral filter kernel diameter")
    parser.add_argument("--bilateral_sigma_color", type=float, default=0.08, help="Bilateral sigmaColor")
    parser.add_argument("--bilateral_sigma_space", type=float, default=7.0, help="Bilateral sigmaSpace")

    # 点云导出
    parser.add_argument("--export_ply", action="store_true", help="Export point cloud as PLY")
    parser.add_argument("--ply_stride", type=int, default=1, help="Subsample stride for point cloud")
    parser.add_argument("--ply_max_points", type=int, default=1200000, help="Maximum number of points in PLY")

    # 相机参数
    parser.add_argument("--fx", type=float, default=None, help="Known fx in pixels")
    parser.add_argument("--fy", type=float, default=None, help="Known fy in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Known cx in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Known cy in pixels")
    parser.add_argument("--fpx", type=float, default=None, help="Known focal length in pixels")
    parser.add_argument("--prefer_known_focal", action="store_true", help="Prefer user-provided focal over model-estimated focal")

    # 尺度校正
    parser.add_argument("--scale", type=float, default=1.0, help="Global depth scale factor")
    parser.add_argument("--bias", type=float, default=0.0, help="Global depth bias in meters")

    # 性能
    parser.add_argument("--fp16", action="store_true", help="Use float16 autocast on CUDA")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_depth_npy(depth: np.ndarray, path: str):
    np.save(path, depth)


def save_depth_vis(depth: np.ndarray, path: str, min_depth: float, max_depth: float):
    depth_vis = np.clip(depth, min_depth, max_depth)
    depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
    depth_vis = (depth_vis * 65535.0).astype(np.uint16)
    Image.fromarray(depth_vis).save(path)


def bilateral_filter_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    d: int = 7,
    sigma_color: float = 0.08,
    sigma_space: float = 7.0,
) -> np.ndarray:
    if not HAS_CV2:
        print("[WARN] OpenCV not installed, skip bilateral filter.")
        return depth

    if valid_mask.sum() == 0:
        return depth

    dmin = depth[valid_mask].min()
    dmax = depth[valid_mask].max()
    if dmax - dmin < 1e-8:
        return depth

    depth_norm = depth.copy()
    depth_norm[valid_mask] = (depth_norm[valid_mask] - dmin) / (dmax - dmin)
    filtered = cv2.bilateralFilter(depth_norm.astype(np.float32), d, sigma_color, sigma_space)

    out = depth.copy()
    out[valid_mask] = filtered[valid_mask] * (dmax - dmin) + dmin
    return out


def clean_depth(depth: np.ndarray, min_depth: float, max_depth: float):
    depth = depth.astype(np.float32)
    valid_mask = np.isfinite(depth)
    valid_mask &= depth > 0

    depth = np.where(valid_mask, depth, 0.0)
    depth = np.clip(depth, min_depth, max_depth)

    valid_mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
    return depth, valid_mask


def build_intrinsics(
    width: int,
    height: int,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    fallback_fpx: float | None,
):
    if fx is None and fy is None and fallback_fpx is not None:
        fx = fallback_fpx
        fy = fallback_fpx

    if fx is None:
        fx = 0.5 * (width + height)
    if fy is None:
        fy = 0.5 * (width + height)

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    return float(fx), float(fy), float(cx), float(cy)


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
    max_points: int = 1200000,
):
    h, w = depth.shape
    ys, xs = np.indices((h, w))

    if stride > 1:
        xs = xs[::stride, ::stride]
        ys = ys[::stride, ::stride]
        depth = depth[::stride, ::stride]
        valid_mask = valid_mask[::stride, ::stride]
        rgb = rgb[::stride, ::stride]

    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    mask = valid_mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
    pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
    cols = rgb[mask].reshape(-1, 3)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    return pts.astype(np.float32), cols.astype(np.uint8)


def save_ply_ascii(points: np.ndarray, colors: np.ndarray, path: str):
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


def extract_model_focal(outputs):
    """
    尝试从 Transformers 输出中提取模型估计焦距
    不同版本字段名可能略有差异，因此做兼容处理
    """
    candidate_names = [
        "focallength_px",
        "focal_length",
        "focal_length_px",
        "estimated_focal_length",
        "field_of_view",
        "fov",
    ]

    for name in candidate_names:
        if hasattr(outputs, name):
            value = getattr(outputs, name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().reshape(-1)[0].item()
            else:
                value = float(value)
            return float(value), name

    if isinstance(outputs, dict):
        for name in candidate_names:
            if name in outputs and outputs[name] is not None:
                value = outputs[name]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().cpu().reshape(-1)[0].item()
                else:
                    value = float(value)
                return float(value), name

    return None, None


def postprocess_depth(processor, outputs, target_size_hw):
    """
    优先尝试 HF 官方 post_process_depth_estimation
    若不可用，则手工插值回原图尺寸
    target_size_hw: (H, W)
    """
    h, w = target_size_hw

    if hasattr(processor, "post_process_depth_estimation"):
        try:
            post = processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(h, w)]
            )
            if isinstance(post, list) and len(post) > 0 and "predicted_depth" in post[0]:
                depth = post[0]["predicted_depth"]
                if isinstance(depth, torch.Tensor):
                    depth = depth.detach().float().cpu().numpy()
                else:
                    depth = np.asarray(depth, dtype=np.float32)
                return depth.astype(np.float32)
        except Exception as e:
            print(f"[WARN] post_process_depth_estimation failed, fallback to interpolate. {e}")

    depth = outputs.predicted_depth
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)  # [B,1,h,w] if needed
    elif depth.ndim == 2:
        depth = depth[None, None, ...]

    depth = torch.nn.functional.interpolate(
        depth,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    depth = depth[0, 0].detach().float().cpu().numpy()
    return depth.astype(np.float32)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    image_path = Path(args.image)
    stem = image_path.stem

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[INFO] Device: {device}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("[INFO] Loading image...")
    image = Image.open(image_path).convert("RGB")
    rgb_np = np.array(image).astype(np.uint8)
    h, w = rgb_np.shape[:2]
    print(f"[INFO] Original size: {w} x {h}")

    print("[INFO] Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_processor
    )
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name)
    model.eval()

    if device == "cuda":
        model = model.to(device)

    print("[INFO] Preparing inputs...")
    inputs = processor(images=image, return_tensors="pt")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    print("[INFO] Running inference...")
    with torch.inference_mode():
        if device == "cuda" and args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    depth = postprocess_depth(processor, outputs, (h, w))

    # 尝试读取模型估计焦距
    model_fpx, focal_name = extract_model_focal(outputs)
    if model_fpx is not None:
        print(f"[INFO] Model-estimated focal from '{focal_name}': {model_fpx}")
    else:
        print("[INFO] No focal output found from model.")

    # 全局尺度修正
    depth = depth * args.scale + args.bias

    # 清理
    depth, valid_mask = clean_depth(depth, args.min_depth, args.max_depth)

    # 边缘保留后处理
    if args.bilateral:
        print("[INFO] Applying bilateral filter...")
        depth = bilateral_filter_depth(
            depth,
            valid_mask,
            d=args.bilateral_d,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
        )
        depth, valid_mask = clean_depth(depth, args.min_depth, args.max_depth)

    # 保存深度
    npy_path = os.path.join(args.output_dir, f"{stem}_depth.npy")
    png_path = os.path.join(args.output_dir, f"{stem}_depth_vis.png")
    save_depth_npy(depth, npy_path)
    save_depth_vis(depth, png_path, args.min_depth, args.max_depth)
    print(f"[INFO] Saved depth numpy: {npy_path}")
    print(f"[INFO] Saved depth visualization: {png_path}")

    # 导出点云
    if args.export_ply:
        # 点云回投优先真实内参
        fallback_fpx = None
        if args.prefer_known_focal:
            fallback_fpx = args.fpx
        else:
            fallback_fpx = args.fpx if args.fpx is not None else model_fpx

        fx, fy, cx, cy = build_intrinsics(
            width=w,
            height=h,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            fallback_fpx=fallback_fpx,
        )

        print("[INFO] Point-cloud intrinsics:")
        print(f"       fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

        points, colors = depth_to_pointcloud(
            depth=depth,
            rgb=rgb_np,
            valid_mask=valid_mask,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=max(1, args.ply_stride),
            max_points=args.ply_max_points,
        )

        ply_path = os.path.join(args.output_dir, f"{stem}.ply")
        save_ply_ascii(points, colors, ply_path)
        print(f"[INFO] Saved point cloud: {ply_path}")
        print(f"[INFO] Point count: {len(points)}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()