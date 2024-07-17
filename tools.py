import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Sampling3D(nn.Module):
    def forward(self, z_mean, z_log_var):

        # 生成标准正态分布的随机噪声
        epsilon = torch.randn_like(z_mean)

        # 计算潜在变量的采样值
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z

class Sampling2D(nn.Module):
    def forward(self, z_mean, z_log_var):
        # 生成标准正态分布的随机噪声
        epsilon = torch.randn_like(z_mean)

        # 计算潜在变量的采样值
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z


class PlanarFlowLayer(nn.Module):
    def __init__(self, z_dim, is_first_layer=True):
        super(PlanarFlowLayer, self).__init__()
        self.z_dim = z_dim
        self.is_first_layer = is_first_layer

        self.w = nn.Parameter(torch.randn(1, self.z_dim), requires_grad=True).to('cuda:0')
        self.u = nn.Parameter(torch.randn(1, self.z_dim), requires_grad=True).to('cuda:0')
        self.b = nn.Parameter(torch.randn(1), requires_grad=True).to('cuda:0')

    def forward(self, inputs):
        EPSILON = 1e-7

        if self.is_first_layer:
            z_prev = inputs
        else:
            z_prev, sum_log_det_jacob = inputs

        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        h = lambda x: torch.tanh(x)
        h_prime = lambda x: 1 - h(x) ** 2

        u_hat = (m(torch.sum(self.w * self.u)) - torch.sum(self.w * self.u)) \
                * (self.w / torch.norm(self.w)) + self.u

        z_prev = z_prev + u_hat * h(torch.sum(z_prev * self.w, dim=-1, keepdim=True) + self.b)

        affine = h_prime(torch.sum(z_prev * self.w, dim=-1, keepdim=True) + self.b) * self.w

        if self.is_first_layer:
            sum_log_det_jacob = torch.log(EPSILON + torch.abs(1 + torch.sum(affine * u_hat, dim=-1)))
        else:
            sum_log_det_jacob += torch.log(EPSILON + torch.abs(1 + torch.sum(affine * u_hat, dim=-1)))

        return z_prev, sum_log_det_jacob


class NFTransformations(nn.Module):
    def __init__(self, z_dim, k):
        super(NFTransformations, self).__init__()
        self.z_dim = z_dim  # 64
        self.k = k   # 8

    def forward(self, z):
        z0 = z
        logD_loss = 0

        zk, logD = PlanarFlowLayer(self.z_dim, True)(z0)

        for i in range(self.k):
            zk, logD = PlanarFlowLayer(self.z_dim, False)((zk, logD))
            logD_loss += logD

        return zk, logD_loss