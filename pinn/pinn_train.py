import sys

sys.path.append(".")
import numpy as np
import torch
from pinn import PINN
import pandas as pd
import time
import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_data(path):
    # Load Data
    file_coordinate = path + "coordinate.csv"
    file_velocity_x = path + "velocity_x.csv"
    file_velocity_y = path + "velocity_y.csv"
    file_velocity_z = path + "velocity_z.csv"

    df_coordinate = pd.read_csv(file_coordinate, header=None)
    df_velocity_x = pd.read_csv(file_velocity_x, header=None)
    df_velocity_y = pd.read_csv(file_velocity_y, header=None)
    df_velocity_z = pd.read_csv(file_velocity_z, header=None)

    array_coordinate = np.array(df_coordinate)
    array_velocity_x = np.array(df_velocity_x)
    array_velocity_y = np.array(df_velocity_y)
    array_velocity_z = np.array(df_velocity_z)
    N = array_coordinate.shape[0]
    S = 1

    # Rearrange Data
    XX = np.tile(array_coordinate[:, 0:1], (1, S))  # N x T
    YY = np.tile(array_coordinate[:, 1:2], (1, S))  # N x T
    ZZ = np.tile(array_coordinate[:, 2:3], (1, S))  # N x T
    ub = np.array([XX.max(), YY.max(), ZZ.max()])
    lb = np.array([XX.min(), YY.min(), ZZ.min()])

    V_X = array_velocity_x.T       # N x T
    V_Y = array_velocity_y.T       # N x T
    V_Z = array_velocity_z.T       # N x T
    ub_velocity = np.array([V_X.max(), V_Y.max(), V_Z.max()])
    lb_velocity = np.array([V_X.min(), V_Y.min(), V_Z.min()])
    print("CSPSV Data size:" + str(V_X.shape))
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    z = ZZ.flatten()[:, None]

    vx = V_X.flatten()[:, None]  # NT x 1
    vy = V_Y.flatten()[:, None]  # NT x 1
    vz = V_Z.flatten()[:, None]  # NT x 1

    return x, y, z, vx, vy, vz, ub, lb, ub_velocity, lb_velocity

# 读取物理点
def load_phy(path):
    file_phy = path + "phy.csv"
    df_phy = pd.read_csv(file_phy, header=None)
    array_phy = np.array(df_phy)

    N = array_phy.shape[0]
    print("Collocaiton size:" + str(N))
    # print(N)
    S = 1
    # Rearrange Data
    XX = np.tile(array_phy[:, 0:1], (1, S))  # N x T
    YY = np.tile(array_phy[:, 1:2], (1, S))  # N x T
    ZZ = np.tile(array_phy[:, 2:3], (1, S))  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    z = ZZ.flatten()[:, None]  # NT x 1

    return x, y, z

def training_data():
    # 读取物理点
    phy_x, phy_y, phy_z = load_phy("../data/")
    phy_xyzs = np.concatenate([phy_x, phy_y, phy_z], axis=1)

    # 读取CSPSV数据
    path_data = "../data/100/"
    x, y, z, vx, vy, vz, ub, lb, ub_velocity, lb_velocity = load_data(path_data)

    xyzs = np.concatenate([x, y, z], axis=1)
    v = np.concatenate([vx, vy, vz], axis=1)

    ######################################################################
    # 物理规律选择点，不放入GPU
    phy_xyzs = torch.tensor(phy_xyzs, dtype=torch.float)
    # 数据点,放入GPU
    xyzs = torch.tensor(xyzs, dtype=torch.float).to(device)
    v = torch.tensor(v, dtype=torch.float).to(device)

    return xyzs, v, ub, lb, ub_velocity, lb_velocity, phy_xyzs

if __name__ == "__main__":
    # 5 x 100
    layers = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]
    data_xyzs, data_v, ub, lb, ub_v, lb_v, phy_xyzs = training_data()
    BATCH_SIZE_PHY = 31000 #107000 #
    pinn = PINN(layers, ub, lb, ub_v, lb_v)

    start_time = time.time()
    dataset_size = phy_xyzs.size()[0]
    iter_batch_count = dataset_size/BATCH_SIZE_PHY
    iter_batch = 0
    for epoch in range(500000):
        # 确定物理规律数据提取位置
        if(iter_batch > iter_batch_count):
            iter_batch = 0
        start = (iter_batch * BATCH_SIZE_PHY) % dataset_size
        end = min(start + BATCH_SIZE_PHY, dataset_size)
        iter_batch += 1
        # 将物理规律数据导入GPU内存
        batch_xyzs_phy = phy_xyzs[start:end].to(device)
        iter, loss,  mse_data, mse_phy = pinn.closure(data_xyzs, data_v, batch_xyzs_phy)
        pinn.adam.step()
        if iter % 50 == 0:
            elapsed = time.time() - start_time
            time_cur = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time_cur + ' Epoch: %d; Iteration: %d; '
                             'Loss: %.3e; (%.3e; %.3e) '
                             'Time consumed: %.2fs; batch size: %d' %
                (epoch, iter, loss, mse_data, mse_phy, elapsed, data_xyzs.size()[0] + batch_xyzs_phy.size()[0]))
            start_time = time.time()
        if iter % 1000 == 0:
            name_model = '../result_model_pinn/model_pinn_%d' % (iter)
            torch.save(pinn, name_model)


