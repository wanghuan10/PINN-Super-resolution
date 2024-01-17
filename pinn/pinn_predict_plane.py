import sys
sys.path.insert(0, './Utilities/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import os
import torch
from mayavi import mlab

PRE_SPEED = 1.03
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 读取txt文件，
# 返回，坐标、速度
def txt_read(filename):
    # ==============开始时间==================
    start = time.perf_counter()
    print(filename + " begins to read!")
    # 判断文件是否存在
    if os.path.exists(filename) == False:
        print("the file is not found!")
    # 判断文件是否为空
    if os.stat(filename).st_size == 0:
        print("the file is empty!")
    with open(filename, encoding='utf-8', errors='ignore') as f_input:
        data = [line.strip().split(',') for line in f_input]
    array_data = np.array(data)
    array_data = np.array(array_data[1:, :], dtype=float)

    # 坐标
    array_xyz = array_data[:, 0:3]
    # 风速
    array_velocity = array_data[:, 3:6]

    # ==============结束时间==================
    end = time.perf_counter()
    print(filename + " is successfully merged! Time consumed: {:.2f}s".format(end - start))
    return array_xyz, array_velocity

def predict_plane(file_model, file_name, fig_predicted, flag, plane=0):
    xyz, velocity = txt_read(file_name)
    if (flag == 2):
        z_thr_lower = abs(xyz[:, 2:3]) >= plane - 0.01
        z_thr_upper = abs(xyz[:, 2:3]) <= plane + 0.01
        z_thr = z_thr_lower & z_thr_upper
        xyz_select_cpu = xyz[z_thr[:, 0]]
        velocity_select = velocity[z_thr[:, 0]]

    if (flag == 3):
        xyz_select_cpu = xyz
        velocity_select = velocity

    # CSPSV训练
    model = torch.load(file_model)
    model.net = model.net.to(device)
    xyz_select = torch.tensor(xyz_select_cpu, dtype=torch.float).to(device)
    u, v, w, p = model.predict(xyz_select)

    vx_predict = u.cpu().detach().numpy()
    vy_predict = v.cpu().detach().numpy()
    vz_predict = w.cpu().detach().numpy()

    return xyz_select_cpu, velocity_select, vx_predict, vy_predict, vz_predict

# 绘制矢量图
# flag：0--x截面；1--y截面；2--z截面
def plane_vector(xyz, velocity, vx, vy, vz, fig_predicted, flag):
    x_predict = xyz[:, 0:1]
    y_predict = xyz[:, 1:2]
    z_predict = xyz[:, 2:3]
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 800))
    if fig_predicted == True:
        mlab.quiver3d(x_predict, y_predict, z_predict, vx, vy, vz)
    else:
        mlab.quiver3d(x_predict, y_predict, z_predict, velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3])
    mlab.view(-40, 90)  # 设置相机的视角 x, z轴视角
    mlab.axes(extent=[0.2, 2.5, 0.2, 1.4, 0, 1.4])
    mlab.axes(xlabel='', ylabel='', zlabel='', line_width=2, nb_labels=6)

    mlab.colorbar(nb_labels=8, label_fmt='%.1f')
    if flag == 3:
        outline_extent = [0.2, 2.5, 0.2, 1.4, 0, 1.4]
    else:
        outline_extent = [0.2, 2.5, 0.2, 1.4, 0, 0]
    mlab.outline(color=(0, 0, 0), opacity=1.0, line_width=2, extent=outline_extent)
    mlab.show()


# 绘制截面图-标量
# flag：0--x截面；1--y截面；2--z截面, 3--3D
def plane_scalar(xyz, velocity, vx, vy, vz, fig_predicted, flag):
    lim_x_lb, lim_x_ub = 0.2, 2.5
    lim_y_lb, lim_y_ub = 0.2, 1.4
    xyz[:, 0:1] = 2.6 - xyz[:, 0:1]
    coordinate = xyz[:, 0:2]
    if fig_predicted == True:
        if flag == 20:
            val = -vx
        if flag == 21:
            val = -vz
    else:
        if flag == 20:
            val = -velocity[:, 0:1]
        if flag == 21:
            val = -velocity[:, 2:3]
    grid_num = 1000

    x = np.linspace(lim_x_lb, lim_x_ub, grid_num)
    y = np.linspace(lim_y_lb, lim_y_ub, grid_num)
    X, Y = np.meshgrid(x, y)
    val_star = griddata(coordinate, val.flatten(), (X, Y), method='linear') # nearest, linear, cubic

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6.5))

    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    ax.imshow(val_star, interpolation='bilinear', cmap=cmap, norm=norm,  # 'rainbow',
              extent=[lim_x_ub, lim_x_lb, lim_y_lb, lim_y_ub],
              origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    # 设置彩虹条字体
    colorbar.ax.tick_params(labelsize=26)
    # 设置坐标刻度字体尺寸
    ax.tick_params(labelsize=26)
    # 设置边框线粗细
    ax.set_aspect('equal', 'box')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.show()

if __name__ == "__main__":
    # 截面标志，0--x截面；1--y截面；2--z截面, 3--3D
    # x: 1.3,; Y: 0.8; Z:0.725
    flag = 3
    plane = 0
    if flag == 0:
        plane = 1.3
    if flag == 1:
        plane = 0.8
    if flag == 2:
        plane = 0.725
    # 提取数据
    file_name = "../data/100/content.txt"
    model_sel = 197000
    file_model = '../result_model_pinn_100/model_pinn_%d' % model_sel
    # 绘制速度场
    fig_predicted = True
    xyz, velocity, vx, vy, vz = predict_plane(file_model, file_name, fig_predicted, flag, plane)
    # flag:
    # cspsv数据绘图--x0， 预测数据绘图x1
    # flag = 20
    # plane_scalar(xyz, velocity, vx, vy, vz, fig_predicted, flag)
    plane_vector(xyz, velocity, vx, vy, vz, fig_predicted, flag)
