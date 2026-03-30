import numpy as np
import open3d as o3d


def world_to_camera_to_pose(R_wc, t_wc):
    """
    已知:
        Xc = R_wc @ Xw + t_wc
    反求:
        相机中心 C_w
        相机到世界旋转 R_cw
    """
    R_cw = R_wc.T
    C_w = -R_cw @ t_wc.reshape(3, 1)
    return R_cw, C_w.reshape(3)


def make_transform(R, t):
    """由 R, t 生成 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def create_camera_axes(R_cw, C_w, axis_size=80.0):
    """
    创建相机自身坐标轴
    相机坐标轴会放置到相机中心，并按 R_cw 旋转到世界系
    """
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    T = make_transform(R_cw, C_w)
    mesh.transform(T)
    return mesh


def create_camera_center_sphere(C_w, radius=8.0):
    """相机中心小球"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(C_w)
    return sphere


def create_optical_axis_arrow(C_w, R_cw, length=120.0, cylinder_radius=2.0, cone_radius=4.0):
    """
    创建相机光轴箭头
    默认取相机坐标系 +Z 方向为光轴方向
    """
    # 相机光轴在世界坐标系中的方向
    optical_dir = R_cw @ np.array([0.0, 0.0, 1.0])
    optical_dir = optical_dir / np.linalg.norm(optical_dir)

    # Open3D 默认箭头沿 +Z 创建
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.compute_vertex_normals()

    # 需要把默认 +Z 方向 旋转到 optical_dir
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, optical_dir)
    c = np.dot(z_axis, optical_dir)

    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            R_align = np.eye(3)
        else:
            # 反向 180°
            R_align = np.array([
                [1, 0,  0],
                [0,-1,  0],
                [0, 0, -1]
            ], dtype=float)
    else:
        vx = np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])
        R_align = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    T = np.eye(4)
    T[:3, :3] = R_align
    T[:3, 3] = C_w
    arrow.transform(T)
    return arrow


def create_line_from_camera_to_target(C_w, R_cw, length=140.0):
    """
    用线段辅助显示光轴方向
    """
    optical_dir = R_cw @ np.array([0.0, 0.0, 1.0])
    optical_dir = optical_dir / np.linalg.norm(optical_dir)
    end_pt = C_w + optical_dir * length

    points = [C_w, end_pt]
    lines = [[0, 1]]
    colors = [[1, 0, 0]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set, end_pt


def create_ground_grid(size=600, step=50):
    """
    在世界坐标系 XY 平面创建网格，帮助观察
    """
    points = []
    lines = []
    colors = []

    idx = 0
    for x in range(-size, size + 1, step):
        points.append([x, -size, 0])
        points.append([x,  size, 0])
        lines.append([idx, idx + 1])
        colors.append([0.7, 0.7, 0.7])
        idx += 2

    for y in range(-size, size + 1, step):
        points.append([-size, y, 0])
        points.append([ size, y, 0])
        lines.append([idx, idx + 1])
        colors.append([0.7, 0.7, 0.7])
        idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=int))
    grid.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=float))
    return grid


def add_camera_visuals(name, R_wc, t_wc, axis_size=80, arrow_length=120):
    """
    为单个相机生成所有可视化对象
    """
    R_cw, C_w = world_to_camera_to_pose(R_wc, t_wc)

    print(f"\n{name}")
    print("=" * 50)
    print("相机中心 C_w =", C_w)
    print("相机朝向矩阵 R_cw =\n", R_cw)
    print("光轴方向(世界系) =", R_cw @ np.array([0.0, 0.0, 1.0]))

    cam_axes = create_camera_axes(R_cw, C_w, axis_size=axis_size)
    cam_center = create_camera_center_sphere(C_w, radius=8.0)
    optical_arrow = create_optical_axis_arrow(C_w, R_cw, length=arrow_length)
    optical_line, end_pt = create_line_from_camera_to_target(C_w, R_cw, length=arrow_length * 1.2)

    return {
        "name": name,
        "C_w": C_w,
        "R_cw": R_cw,
        "geometries": [cam_axes, cam_center, optical_arrow, optical_line],
        "look_point": end_pt
    }


def main():
    # ============================================================
    # 把这里替换成你的三台相机 world->camera 标定结果
    # Xc = R_wc @ Xw + t_wc
    # ============================================================





    R1_wc = np.array([
        [0.5547, 0.8320, 3.3764e-04],
        [-0.0066, 0.0040, 1],
        [0.8320, -0.5547, 0.0077]
    ], dtype=np.float64)
    t1_wc = np.array(  [-16.2687, -0.0172, 260.7818], dtype=np.float64)

    R2_wc = np.array([
        [0.3875, -0.9218, 0.0040],
        [0.6631,  0.2818, 0.6934],
        [-0.6404, -0.2661,0.7205]
    ], dtype=np.float64)
    t2_wc = np.array( [1.9990, -40.0983, 210.5670], dtype=np.float64)


    R3_wc = np.array([
        [-0.9988,  0.0474, -0.0084],
        [ -0.0188, -0.2240, 0.9744],
         [ 0.0443,  0.9734, 0.2247]
    ], dtype=np.float64)
    t3_wc = np.array(   [-2.6113, -4.5020, 264.5247], dtype=np.float64)

    # ============================================================
    # 世界坐标系
    # ============================================================
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=120.0, origin=[0, 0, 0])

    # 地面网格
    grid = create_ground_grid(size=600, step=50)

    # 三台相机
    cam1 = add_camera_visuals("Camera_1", R1_wc, t1_wc, axis_size=70, arrow_length=120)
    cam2 = add_camera_visuals("Camera_2", R2_wc, t2_wc, axis_size=70, arrow_length=120)
    cam3 = add_camera_visuals("Camera_3", R3_wc, t3_wc, axis_size=70, arrow_length=120)

    geometries = [world_axes, grid]
    geometries.extend(cam1["geometries"])
    geometries.extend(cam2["geometries"])
    geometries.extend(cam3["geometries"])

    # 额外加一个世界原点小球，方便看
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
    origin_sphere.compute_vertex_normals()
    origin_sphere.translate([0, 0, 0])
    geometries.append(origin_sphere)

    # 启动可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Pose Viewer", width=1400, height=900)
    for g in geometries:
        vis.add_geometry(g)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    render_option.line_width = 2.0
    render_option.point_size = 5.0

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.3, -0.4, -0.8])
    view_ctl.set_lookat([0, 0, 200])
    view_ctl.set_up([0, 0, 1])
    view_ctl.set_zoom(0.7)

    print("\n使用说明：")
    print("1. 红/绿/蓝 = X/Y/Z 轴")
    print("2. 世界坐标系在原点")
    print("3. 每台相机位置处有一个小球")
    print("4. 每台相机前方的箭头表示光轴朝向（默认相机 +Z 方向）")
    print("5. 如果你的系统中相机光轴定义是 -Z，可把下面代码里的 [0,0,1] 改成 [0,0,-1]")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()