import numpy as np
import pyvista as pv
from scipy.io import loadmat
from pcc_Kinematic import nodes_piecewise_cc

# ========== 1. 读取 节点和单元 ==========

nodes_mat = loadmat("nodes_coord.mat")
nodes = nodes_mat["nodes_coord"]

tets_mat = loadmat("tet_connect.mat")
tets = tets_mat["tet_connect"].astype(int)

# 节点形状处理
if nodes.shape[1] != 3 and nodes.shape[0] == 3:
    nodes = nodes.T  # (N,3)
elif nodes.shape[1] != 3 and nodes.shape[0] != 3:
    raise ValueError(f"nodes_coord 的形状是 {nodes.shape}，请检查。")

# 单元形状处理
if tets.shape[1] != 4 and tets.shape[0] == 4:
    tets = tets.T  # (M,4)
elif tets.shape[1] != 4 and tets.shape[0] != 4:
    raise ValueError(f"tet_connect 的形状是 {tets.shape}，请检查。")

# MATLAB -> Python 索引
if tets.min() == 1:
    tets = tets - 1


# ========== 2. 读取位移，并生成变形后坐标 ==========

# disp_mat = loadmat("x338_full.mat")   # 自己的位移文件名
# u = disp_mat["x338_full"].reshape([-1, 3])    
disp_mat = loadmat("x440_full_a.mat")   # 自己的位移文件名
u = disp_mat["x440_full_a"].reshape([-1, 3])    

# 形状处理：保证 (N,3)
if u.shape[1] != 3 and u.shape[0] == 3:
    u = u.T
elif u.shape[1] != 3 and u.shape[0] != 3:
    raise ValueError(f"disp 的形状是 {u.shape}，请检查。")

if u.shape[0] != nodes.shape[0]:
    raise ValueError(f"位移节点数 {u.shape[0]} 与 坐标节点数 {nodes.shape[0]} 不一致。")

# 位移放大因子（一般有限元位移很小，适当放大看看形状）
scale = 1.0  # 比如 10 或 100，按需要改

nodes_def = nodes + scale * u        # 变形后坐标 (N,3)

# 位移模长 |u|
u_mag = np.linalg.norm(u, axis=1)    # (N,)

# ========== 可选：定义一个相机放大视图的函数 ==========
def zoom_view(plotter, factor=1.5):
    """
    相机放大视图：
    factor > 1  放大
    factor < 1  缩小
    """
    pos, focal, viewup = plotter.camera_position
    pos = np.array(pos)
    focal = np.array(focal)

    # 当前相机到焦点的向量
    v = pos - focal
    # 缩短这段距离，相当于把相机拉近
    new_pos = focal + v / factor

    plotter.camera_position = [tuple(new_pos), tuple(focal), tuple(viewup)]

# ========== 3. 提取外表面（函数和之前一样） ==========

def extract_surface_from_tets(points, tets):
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 1, 3]]
    f2 = tets[:, [0, 2, 3]]
    f3 = tets[:, [1, 2, 3]]

    faces = np.vstack([f0, f1, f2, f3])  # (4M,3)
    faces_sorted = np.sort(faces, axis=1)

    uniq_faces, idx, counts = np.unique(
        faces_sorted, axis=0, return_index=True, return_counts=True
    )

    boundary_faces = uniq_faces[counts == 1]  # (K,3)

    n_tri = boundary_faces.shape[0]
    faces_pv = np.hstack(
        [np.full((n_tri, 1), 3, dtype=np.int64), boundary_faces.astype(np.int64)]
    ).ravel()

    surf = pv.PolyData(points, faces_pv)
    return surf

# 未变形表面（可选，用来对比）
surf_undeformed = extract_surface_from_tets(nodes, tets)

# 变形后表面
surf_deformed = extract_surface_from_tets(nodes_def, tets)

# ========== 4. 对变形后表面做平滑，并挂上位移模长标量 ==========

surf_def_smooth = surf_deformed.smooth(
    n_iter=50,
    relaxation_factor=0.02,
    feature_angle=60.0
)

# 注意：smooth 不改变点数，因此可以直接用 u_mag
surf_def_smooth["u_mag"] = u_mag*1000   # 节点标量 = 位移模长

mesh_flipped = surf_def_smooth.copy()
mesh_flipped.points[:, 2] *= -1  # 把 z 轴取负，实现上下翻转
mesh_flipped.points[:, 0] *= -1  # 把 x 轴取负，实现上下翻转

mesh_unformed_flipped = surf_undeformed.copy()
mesh_unformed_flipped.points[:, 2] *= -1  # 把 z 轴取负，实现上下翻转
mesh_unformed_flipped.points[:, 0] *= -1  # 把 x 轴取负，实现上下翻转

# ========== 额外：创建并翻转球面 ==========
sphere_center = np.array([0.0088+0.003, 0.0820+0.02, 0.3519], dtype=float)  # (m)
sphere_radius = 0.025  # (m)

# 由于你把所有网格 z *= -1 翻转了，所以球也要翻转：把圆心 z 取负
sphere_center_flipped = sphere_center.copy()
sphere_center_flipped[2] *= -1 # z 取负, 
sphere_center_flipped[0] *= -1 # z 取负, 

sphere = pv.Sphere(
    radius=sphere_radius,
    center=sphere_center_flipped,
    theta_resolution=32,
    phi_resolution=32
)

# ========== 5. 绘制：按位移大小着色的变形结构 ==========

p = pv.Plotter(window_size=(1080, 1920))

# 可选：先画一个淡淡的未变形轮廓，方便对比
p.add_mesh(
    # surf_undeformed,
    mesh_unformed_flipped,
    show_edges=False,
    smooth_shading=True,
    opacity=0.1,
    color="grey"
)

# 再画变形后的结构，并按位移模长上色
p.add_mesh(
    # surf_def_smooth,
    mesh_flipped,
    scalars="u_mag",
    show_edges=True,        # 不画边线，只显示光滑色片
    smooth_shading=True,
    opacity=0.85,
    cmap="turbo",          # colormap，可换成 "jet"、"turbo"、"viridis" 等
    scalar_bar_args=dict(
        title="Displacement \n magnitude (mm)",
        vertical=True,
        position_x=0.7,    # 0~1：越大越靠右
        position_y=0.1,     # 0~1：越大越靠上
        width=0.3,          # 宽度（相对窗口）
        height=0.6,         # 高度（相对窗口）
        title_font_size=28,
        label_font_size=22,
        n_labels=5,            # 可选：减少刻度数量避免拥挤
    )
)

# ========== 绘制球面（半透明） ==========
p.add_mesh(
    sphere,
    color="orange",
    opacity=1,
    smooth_shading=True,
    show_edges=False
)

# 如果你想让球边缘更明显，可以加一圈线框：
p.add_mesh(
    sphere,
    color="black",
    opacity=1,
    style="wireframe",
    line_width=2
)


# # 加到已有的绘图里
# p.add_mesh(
#     curve_copy,
#     color="grey",        # 曲线颜色
#     line_width=10,       # 线宽
# )
# p.camera_position = "yx"  # xz 视角
p.camera_position = "yz"  # xz 视角
# p.camera_position = "xz"  # xz 视角
# p.view_isometric()   # 3D 等轴视角



p.show_axes = True
zoom_view(p, factor=2)
# p.set_aspect("equal")
p.enable_anti_aliasing('fxaa')   # 或 'fxaa' 'ssaa'
# p.enable_eye_dome_lighting()     # 点云/表面细节更立体（可选）

p.show()
p.screenshot("SnakeDeformed_any2.png", scale=5.0)  # 分辨率放大 2 倍
# p.save_graphic("SnakeDeformed.eps")   # 或 .eps, .pdf（视环境支持）

# mesh_flipped.save("deformed_surface.obj")  # VTK PolyData
# 或者 .ply, .obj, .stl 等

p.close()