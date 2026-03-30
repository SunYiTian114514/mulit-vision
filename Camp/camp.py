import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# 数学部分
# ============================================================

def world_to_camera_to_pose(R_wc, t_wc):
    """
    已知:
        Xc = R_wc @ Xw + t_wc
    求:
        R_cw: 相机坐标系 -> 世界坐标系
        C_w : 相机中心在世界坐标系中的位置
    """
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3, 1)

    R_cw = R_wc.T
    C_w = -R_cw @ t_wc
    return R_cw, C_w.reshape(3)


def normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# ============================================================
# 绘图辅助
# ============================================================

def set_axes_equal(ax):
    """
    让 3D 坐标轴比例一致，避免相机形状被拉伸
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def draw_world_axes(ax, origin=np.zeros(3), length=80):
    o = np.asarray(origin, dtype=np.float64)
    ax.quiver(o[0], o[1], o[2], length, 0, 0, color='r', linewidth=2.2, arrow_length_ratio=0.08)
    ax.quiver(o[0], o[1], o[2], 0, length, 0, color='g', linewidth=2.2, arrow_length_ratio=0.08)
    ax.quiver(o[0], o[1], o[2], 0, 0, length, color='b', linewidth=2.2, arrow_length_ratio=0.08)
    ax.text(o[0], o[1], o[2] + 10, "World", fontsize=11, weight='bold')


def draw_ground_grid(ax, size=300, step=30, z=0.0):
    vals = np.arange(-size, size + step, step)

    for x in vals:
        lw = 1.1 if abs(x) < 1e-9 else 0.6
        color = '0.55' if abs(x) < 1e-9 else '0.82'
        ax.plot([x, x], [-size, size], [z, z], color=color, linewidth=lw, zorder=0)

    for y in vals:
        lw = 1.1 if abs(y) < 1e-9 else 0.6
        color = '0.55' if abs(y) < 1e-9 else '0.82'
        ax.plot([-size, size], [y, y], [z, z], color=color, linewidth=lw, zorder=0)


def draw_camera_axes(ax, C, R_cw, axis_len=28):
    """
    相机局部坐标轴
    R_cw 的三列分别是相机 x/y/z 轴在世界系下的方向
    """
    C = np.asarray(C, dtype=np.float64)
    x_axis = R_cw[:, 0]
    y_axis = R_cw[:, 1]
    z_axis = R_cw[:, 2]

    ax.quiver(C[0], C[1], C[2], x_axis[0], x_axis[1], x_axis[2],
              length=axis_len, color='r', linewidth=1.6, arrow_length_ratio=0.12)
    ax.quiver(C[0], C[1], C[2], y_axis[0], y_axis[1], y_axis[2],
              length=axis_len, color='g', linewidth=1.6, arrow_length_ratio=0.12)
    ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2],
              length=axis_len, color='b', linewidth=1.6, arrow_length_ratio=0.12)


def draw_camera_center(ax, C, color='k', size=35):
    C = np.asarray(C, dtype=np.float64)
    ax.scatter([C[0]], [C[1]], [C[2]], color=color, s=size, depthshade=False, zorder=5)


def draw_optical_axis(ax, C, R_cw, length=55, color='crimson', optical_axis_sign=+1):
    C = np.asarray(C, dtype=np.float64)
    optical_dir = R_cw @ np.array([0.0, 0.0, float(optical_axis_sign)])
    optical_dir = normalize(optical_dir)

    ax.quiver(C[0], C[1], C[2],
              optical_dir[0], optical_dir[1], optical_dir[2],
              length=length, color=color, linewidth=2.3, arrow_length_ratio=0.14)

    return optical_dir


def draw_frustum(ax, C, R_cw, scale=42, fov_x_deg=52, fov_y_deg=40,
                 color='tab:blue', alpha=0.10, lw=1.5):
    """
    画相机视锥体
    默认光轴为相机 +Z
    """
    C = np.asarray(C, dtype=np.float64)

    fx = np.tan(np.deg2rad(fov_x_deg / 2.0))
    fy = np.tan(np.deg2rad(fov_y_deg / 2.0))
    z = scale

    corners_cam = np.array([
        [-fx * z, -fy * z, z],
        [ fx * z, -fy * z, z],
        [ fx * z,  fy * z, z],
        [-fx * z,  fy * z, z]
    ], dtype=np.float64)

    corners_world = (R_cw @ corners_cam.T).T + C.reshape(1, 3)

    # 线框
    for p in corners_world:
        ax.plot([C[0], p[0]], [C[1], p[1]], [C[2], p[2]], color=color, linewidth=lw)

    for i in range(4):
        p1 = corners_world[i]
        p2 = corners_world[(i + 1) % 4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=lw)

    # 前端半透明面
    verts = [corners_world]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(poly)

    return corners_world


def annotate_camera(ax, name, C, optical_dir, text_offset=12):
    C = np.asarray(C, dtype=np.float64)
    text_pos = C + np.array([0.0, 0.0, text_offset])

    ax.text(text_pos[0], text_pos[1], text_pos[2],
            name, fontsize=11, weight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.9))

    coord_pos = C + np.array([8.0, 8.0, -8.0])
    ax.text(coord_pos[0], coord_pos[1], coord_pos[2],
            f"({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f})",
            fontsize=8.5, color='0.25')


def plot_camera(ax, name, R_wc, t_wc, cam_color='tab:blue', optical_axis_sign=+1):
    R_cw, C = world_to_camera_to_pose(R_wc, t_wc)

    draw_camera_center(ax, C, color='k', size=34)
    draw_camera_axes(ax, C, R_cw, axis_len=26)
    optical_dir = draw_optical_axis(ax, C, R_cw, length=58, color='crimson',
                                    optical_axis_sign=optical_axis_sign)
    draw_frustum(ax, C, R_cw, scale=40, color=cam_color, alpha=0.10, lw=1.6)
    annotate_camera(ax, name, C, optical_dir)

    return {
        "name": name,
        "C_w": C,
        "R_cw": R_cw,
        "optical_dir": optical_dir
    }


# ============================================================
# 主程序
# ============================================================

def main():
    # ========================================================
    # 你的相机外参，格式:
    # Xc = R_wc @ Xw + t_wc
    # ========================================================

    R1_wc = np.array([
        [0.5547, 0.8320, 3.3764e-04],
        [-0.0066, 0.0040, 1],
        [0.8320, -0.5547, 0.0077]
    ], dtype=np.float64)
    t1_wc = np.array([-16.2687, -0.0172, 260.7818], dtype=np.float64)

    R2_wc = np.array([
        [0.3875, -0.9218, 0.0040],
        [0.6631, 0.2818, 0.6934],
        [-0.6404, -0.2661, 0.7205]
    ], dtype=np.float64)
    T2_wc = np.array([1.9990, -40.0983, 210.5670], dtype=np.float64)

    R3_wc = np.array([
        [-0.9988, 0.0474, -0.0084],
        [-0.0188, -0.2240, 0.9744],
        [0.0443, 0.9734, 0.2247]
    ], dtype=np.float64)
    T3_wc = np.array([-2.6113, -4.5020, 264.5247], dtype=np.float64)

    # 如果你的光轴定义是 -Z，把下面改成 -1
    optical_axis_sign = +1

    fig = plt.figure(figsize=(11, 9), dpi=160)
    ax = fig.add_subplot(111, projection='3d')

    # 白底科研风
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 去掉 pane 的重颜色
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    # 先画地面和世界系
    draw_ground_grid(ax, size=300, step=30, z=0.0)
    draw_world_axes(ax, origin=np.array([0.0, 0.0, 0.0]), length=75)

    cam_infos = []
    cam_infos.append(plot_camera(ax, "Cam1", R1_wc, t1_wc, cam_color='tab:blue',
                                 optical_axis_sign=optical_axis_sign))
    cam_infos.append(plot_camera(ax, "Cam2", R2_wc, t2_wc, cam_color='tab:green',
                                 optical_axis_sign=optical_axis_sign))
    cam_infos.append(plot_camera(ax, "Cam3", R3_wc, t3_wc, cam_color='tab:purple',
                                 optical_axis_sign=optical_axis_sign))

    # 可选：连线显示相机阵列关系
    centers = np.array([c["C_w"] for c in cam_infos])
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            linestyle='--', linewidth=1.2, color='0.45', alpha=0.8)

    # 坐标轴标题
    ax.set_xlabel("X / world", labelpad=10, fontsize=11)
    ax.set_ylabel("Y / world", labelpad=10, fontsize=11)
    ax.set_zlabel("Z / world", labelpad=10, fontsize=11)

    # 标题
    ax.set_title("Multi-Camera Pose Visualization", fontsize=14, pad=18, weight='bold')

    # 视角更像科研图
    ax.view_init(elev=24, azim=-58)

    # 自动范围
    all_pts = [np.array([0.0, 0.0, 0.0])]
    for cam in cam_infos:
        all_pts.append(cam["C_w"])
        all_pts.append(cam["C_w"] + cam["optical_dir"] * 60.0)

    all_pts = np.array(all_pts)
    margin = 60.0

    ax.set_xlim(np.min(all_pts[:, 0]) - margin, np.max(all_pts[:, 0]) + margin)
    ax.set_ylim(np.min(all_pts[:, 1]) - margin, np.max(all_pts[:, 1]) + margin)
    ax.set_zlim(np.min(all_pts[:, 2]) - margin, np.max(all_pts[:, 2]) + margin)

    set_axes_equal(ax)
    plt.tight_layout()

    # 保存高清图
    plt.savefig("camera_pose_scientific.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 控制台输出
    print("\n" + "=" * 70)
    print("Camera Pose Summary")
    print("=" * 70)
    for cam in cam_infos:
        print(f"\n{cam['name']}")
        print("-" * 40)
        print("Camera center C_w:")
        print(np.array2string(cam["C_w"], precision=4, suppress_small=True))
        print("Optical direction:")
        print(np.array2string(cam["optical_dir"], precision=4, suppress_small=True))
        print("R_cw:")
        print(np.array2string(cam["R_cw"], precision=4, suppress_small=True))


if __name__ == "__main__":
    main()