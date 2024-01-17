import sys

sys.path.append(".")
import torch
from torch.autograd import grad
from network import DNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PINN:
    def __init__(self, layers, ub, lb, ub_stem, lb_stem):
        self.ub_stem = ub_stem
        self.lb_stem = lb_stem
        self.alpha = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1x = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1y = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1z = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2x = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2y = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2z = torch.tensor([0.0], requires_grad=True).to(device)
        self.alpha = torch.nn.Parameter(self.alpha)
        self.lambda_1x = torch.nn.Parameter(self.lambda_1x)
        self.lambda_1y = torch.nn.Parameter(self.lambda_1y)
        self.lambda_1z = torch.nn.Parameter(self.lambda_1z)
        self.lambda_2x = torch.nn.Parameter(self.lambda_2x)
        self.lambda_2y = torch.nn.Parameter(self.lambda_2y)
        self.lambda_2z = torch.nn.Parameter(self.lambda_2z)
        self.net = DNN(layers, ub=ub, lb=lb).to(device)
        self.net.register_parameter("alpha", self.alpha)
        self.net.register_parameter("lambda_1x", self.lambda_1x)
        self.net.register_parameter("lambda_1y", self.lambda_1y)
        self.net.register_parameter("lambda_1z", self.lambda_1z)
        self.net.register_parameter("lambda_2x", self.lambda_2x)
        self.net.register_parameter("lambda_2y", self.lambda_2y)
        self.net.register_parameter("lambda_2z", self.lambda_2z)
        self.adam = torch.optim.Adam(self.net.parameters())
        self.iter = 0

    def predict(self, xyzts):
        if xyzts.requires_grad == False:
            xyzts = xyzts.clone()
            xyzts.requires_grad = True
        psi_p_t = self.net(xyzts)
        u = psi_p_t[:, 0:1]
        v = psi_p_t[:, 1:2]
        w = psi_p_t[:, 2:3]
        p = psi_p_t[:, 3:4]

        return u, v, w, p

    def loss_func(self, xyz, vtem, phy_xyz):
        phy_xyz = phy_xyz.clone()
        phy_xyz.requires_grad = True
        alpha = self.alpha
        lambda_1x = self.lambda_1x
        lambda_1y = self.lambda_1y
        lambda_1z = self.lambda_1z
        lambda_2x = self.lambda_2x
        lambda_2y = self.lambda_2y
        lambda_2z = self.lambda_2z

        # 数据
        u1, v1, w1, p1 = self.predict(xyz)
        ub = self.ub_stem
        lb = self.lb_stem
        weight_u = 1
        weight_v = 1
        weight_w = 1

        u_r = u1 #/ uvw_sum
        v_r = v1 #/ uvw_sum
        w_r = w1 #/ uvw_sum
        u_r = torch.where(torch.isnan(u_r), torch.full_like(u_r, 0), u_r)
        v_r = torch.where(torch.isnan(v_r), torch.full_like(v_r, 0), v_r)
        w_r = torch.where(torch.isnan(w_r), torch.full_like(w_r, 0), w_r)

        u_o = vtem[:, 0:1]
        v_o = vtem[:, 1:2]
        w_o = vtem[:, 2:3]
        u_o_r = u_o #/ uvw_o_sum
        v_o_r = v_o #/ uvw_o_sum
        w_o_r = w_o #/ uvw_o_sum
        u_o_r = torch.where(torch.isnan(u_o_r), torch.full_like(u_o_r, 0), u_o_r)
        v_o_r = torch.where(torch.isnan(v_o_r), torch.full_like(v_o_r, 0), v_o_r)
        w_o_r = torch.where(torch.isnan(w_o_r), torch.full_like(w_o_r, 0), w_o_r)

        # MRE
        mse_data = weight_u * torch.sum(torch.abs((u_r - u_o_r) / u_o_r)) + \
                   weight_v * torch.sum(torch.abs((v_r - v_o_r) / v_o_r)) + \
                   weight_w * torch.sum(torch.abs((w_r - w_o_r) / w_o_r))

        # 物理规律
        u, v, w, p = self.predict(phy_xyz)
        # X轴
        u_xyzt = grad(u.sum(), phy_xyz, create_graph=True)[0]
        u_x = u_xyzt[:, 0:1]
        u_y = u_xyzt[:, 1:2]
        u_z = u_xyzt[:, 2:3]

        u_xx = grad(u_x.sum(), phy_xyz, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), phy_xyz, create_graph=True)[0][:, 1:2]
        u_zz = grad(u_z.sum(), phy_xyz, create_graph=True)[0][:, 2:3]
        u_yx = grad(u_y.sum(), phy_xyz, create_graph=True)[0][:, 0:1]
        u_zx = grad(u_z.sum(), phy_xyz, create_graph=True)[0][:, 0:1]
        # y轴
        v_xyzt = grad(v.sum(), phy_xyz, create_graph=True)[0]
        v_x = v_xyzt[:, 0:1]
        v_y = v_xyzt[:, 1:2]
        v_z = v_xyzt[:, 2:3]

        v_xx = grad(v_x.sum(), phy_xyz, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), phy_xyz, create_graph=True)[0][:, 1:2]
        v_zz = grad(v_z.sum(), phy_xyz, create_graph=True)[0][:, 2:3]
        v_xy = grad(v_x.sum(), phy_xyz, create_graph=True)[0][:, 1:2]
        v_zy = grad(v_z.sum(), phy_xyz, create_graph=True)[0][:, 1:2]
        # z轴
        w_xyzt = grad(w.sum(), phy_xyz, create_graph=True)[0]
        w_x = w_xyzt[:, 0:1]
        w_y = w_xyzt[:, 1:2]
        w_z = w_xyzt[:, 2:3]

        w_xx = grad(w_x.sum(), phy_xyz, create_graph=True)[0][:, 0:1]
        w_yy = grad(w_y.sum(), phy_xyz, create_graph=True)[0][:, 1:2]
        w_zz = grad(w_z.sum(), phy_xyz, create_graph=True)[0][:, 2:3]
        w_xz = grad(w_x.sum(), phy_xyz, create_graph=True)[0][:, 2:3]
        w_yz = grad(w_y.sum(), phy_xyz, create_graph=True)[0][:, 2:3]

        # 压力
        p_xyzt = grad(p.sum(), phy_xyz, create_graph=True)[0]
        p_x = p_xyzt[:, 0:1]
        p_y = p_xyzt[:, 1:2]
        p_z = p_xyzt[:, 2:3]

        # 连续方程
        f_c = u_x + v_y + w_z

        # 动量方程
        f_u = u * u_x + v * u_y + w * u_z + lambda_1x * p_x - lambda_2x * (u_xx + u_yy + u_zz + u_xx + v_xy + w_xz)
        f_v = u * v_x + v * v_y + w * v_z + lambda_1y * p_y - lambda_2y * (v_xx + v_yy + v_zz + u_yx + v_yy + w_yz)
        f_w = u * w_x + v * w_y + w * w_z + lambda_1z * p_z - lambda_2z * (w_xx + w_yy + w_zz + u_zx + v_zy + w_zz)

        mse_phy = torch.sum(torch.square(f_c)) + \
                torch.sum(torch.square(f_u)) + torch.sum(torch.square(f_v)) + torch.sum(torch.square(f_w))
        weight_data = 5
        weight_phy = 1
        loss = mse_data * weight_data + mse_phy * weight_phy
        return loss, mse_data, mse_phy

    def closure(self, xyzts, v, phy_xyzs):
        self.adam.zero_grad()

        loss, mse_v, mse_f = self.loss_func(xyzts, v, phy_xyzs)
        loss.backward()

        self.iter += 1

        return self.iter, loss, mse_v, mse_f
