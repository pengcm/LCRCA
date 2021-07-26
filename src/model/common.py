import math

import torch
import torch.nn as nn
# import torch.nn.functional as F



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d( in_channels, out_channels, kernel_size,
                        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=True, res_scale=1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if (i == 0 and act):
                m.append(nn.ReLU(True))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)          # 加权权重
        return res + x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act == 'relu':
            m.append(nn.ReLU(True))
        elif act == 'prelu':
            m.append(nn.PReLU(n_feats))

        super(Upsampler, self).__init__(*m)





####============================================================

def _pdf(x):
    x_mean  = torch.mean(x,dim=(-2,-1),keepdim=True)
    x_bias  = x - x_mean
    # x_k1    =  x_bias.abs().mean(dim=(-2,-1),keepdim=True)
    x_k2 =  x_bias.pow(2).mean(dim=(-2,-1),keepdim=True)  # 二阶中心矩
    # x_k3 = x_bias.pow(3).mean(dim=(-2,-1),keepdim=True)
    x_k4  =  x_bias.pow(4).mean(dim=(-2,-1),keepdim=True)

    x_sigma = x_k2.pow(1/2)      # 为了统一量纲
    # # x_skew = x_k3 / x_sigma.pow(1.5)        # 偏度 开3次方
    x_kurt = x_k4.pow(1/4) / x_sigma       # 峰度 开4次方

    return torch.cat([x_mean,x_sigma,x_kurt,],dim=-1)

def _spatialpool(x):
    x_avg = torch.mean(x, dim=1,keepdim=True)
    x_max = torch.max( x, dim=1,keepdim=True)[0]

    return torch.cat([x_avg, x_max], dim=1)

## Channel Attention (CA) Layer  ----from RCAN
class SCALayer(nn.Module):
    def __init__(self, n_feats, bias=True, reduction=16, increase=2):
        super(SCALayer, self).__init__()

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.f = _pdf
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feats, 4, (1,1), padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, n_feats, (1,3), padding=0, bias=bias),
            nn.Sigmoid(),
        )
        # self.sptialpool = _spatialpool
        # self.conv_space = nn.Sequential(
        #     # nn.Conv2d(n_feats, 2*n_feats, 1, padding=0, bias=bias),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(1, 1, 3, padding=1, bias=bias),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(2, 1, 3, padding=1, bias=bias),
        #     nn.Sigmoid(),
        # )
        # self.out = nn.Conv2d(2*n_feats,n_feats,1,1,0,bias=bias )
        # self.coef = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0, 1.0]))


    def forward(self, x):
        # print(self.coef)
        y = self.f(x)
        y = self.conv_du(y)        #   N,C,1,1
        y = x*y

        # z = self.sptialpool(x)
        # z = self.conv_space(z) - 0.5
        # z = x*z*self.coef[3]
        # # out = self.out(torch.cat([y,z],dim=1))
        # t = self.coef[0] + self.coef[1] 
        return y


####============================================================
# class LightResBlock(nn.Module):
#     def __init__(self, conv, n_feats, kernel_size,
#                 bias , redu=2):
#         super(LightResBlock, self).__init__()

#         self.conv1 = nn.Sequential(
#             # conv(n_feats, n_feats//redu,1,bias=bias),
#             # nn.ReLU(True),
#             conv(n_feats, n_feats,kernel_size,bias=bias),
#             nn.ReLU(True),
#         )
#         self.conv2 = nn.Sequential(
#             # conv(n_feats, n_feats//redu,1,bias=bias),
#             # nn.ReLU(True),
#             conv(n_feats, n_feats,kernel_size,bias=bias),
#         )
#         self.spa = nn.Sequential(
#             conv(n_feats, n_feats*redu,1,bias=bias),
#             nn.ReLU(True),
#             conv(n_feats*redu, 1,1,bias=bias),
#             nn.Sigmoid(),
#         )

#     def forward(self,x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1 + x)
#         y = self.spa(x2)

#         return x + x2*y

class LightResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                bias , relu_a=0.2, redu=2):
        super(LightResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            conv(n_feats, n_feats,kernel_size,bias=bias),
            nn.ReLU(True),
            conv(n_feats, n_feats,kernel_size,bias=bias),
            # nn.ReLU(True),
        )
        # self.conv2 = nn.Sequential(
        #     conv(n_feats, n_feats//redu,kernel_size,bias=bias),
        #     # nn.PReLU(n_feats//redu,relu_a),
        #     nn.ReLU(True),
        #     conv(n_feats//redu, n_feats,kernel_size,bias=bias),
        # )

    def forward(self,x):
        x1 = self.conv1(x)
        # x1 = self.conv2(x1 + x)
        return x + x1
        # x = self.cca(x)
        # return x + self.conv3(x)

class ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias):
        super(ResGroup, self).__init__()

        self.b1 = LightResBlock(conv, n_feats, kernel_size, bias)
        self.b2 = LightResBlock(conv, n_feats, kernel_size, bias)
        self.b3 = LightResBlock(conv, n_feats, kernel_size, bias)
        self.cat1 = conv( 2 * n_feats, n_feats, 1, bias)
        self.cat2 = conv( 3 * n_feats, n_feats, 1, bias)
        self.cca = SCALayer(n_feats, )


    def forward(self, x):
        x1 = self.b1(x)
        cat1 = torch.cat([x,x1],dim=1)
        x2 = self.b2( self.cat1( cat1))
        x3 = self.b3( self.cat2( torch.cat([cat1, x2], dim=1)))
        x1 = self.cca(x1 + x2 +x3)
        return x1

####============================================================

class UpInter(nn.Sequential):
    def __init__(self, conv, scale, n_in, n_out=3, bias=True):
        m = []
        m.append( conv(n_in, scale * scale* n_out, 3, bias=bias))
        m.append( nn.PixelShuffle(scale))
        super(UpInter, self).__init__(*m)




####################################-----------------------------------
# class ResBlock(nn.Module):
#     def __init__(self, n_feats=64, # kernel_size,conv
#                 bias=True, bn=False, act=True, res_scale=0.1):

#         super(ResBlock, self).__init__()

#         self.pre = nn.Sequential(
#             nn.Conv2d(n_feats,n_feats, 3,1,1),
#             nn.ReLU(True),
#         )
#         self.conv1  = nn.ModuleList(
#             nn.Conv2d(n_feats,n_feats, 1,1,0) for _ in range(2)
#         )

#         self.conv3 = nn.ModuleList(
#             nn.Conv2d(n_feats//4, n_feats//4, 3,1,1) for _ in range(3)
#         )

#         self.res_scale = res_scale

#     def forward(self,x):
#         y0 = self.pre(x)
#         y0 = self.conv1[0](x)
#         y0,y1,y2,y3 = torch.chunk(y0,4,1)
#         y1 = self.conv3[0](y1)
#         y2 = self.conv3[1](y1 + y2)
#         y3 = self.conv3[2](y2 + y3)
#         y0 = torch.cat([y0,y1,y2,y3], dim=1)
#         y0 = self.conv1[1](y0)
#         return x + y0.mul(self.res_scale)




####################################-----------------------------------

# class DepDenseBlock( nn.Module):
#     ''' Dense Net consist of DepResBlock
#       input size: '''
#     def __init__(self, n_feats=64, conv=default_conv):
#         super( DepDenseBlock, self).__init__()

#         self.DDB1 = DepResBlock( conv, n_feats,)
#         self.DDB2 = DepResBlock( conv, n_feats * 2,)
#         self.DDB3 = DepResBlock( conv, n_feats * 4,)
#         self.out = nn.Sequential(
#             conv( n_feats*8, n_feats, 1 ),
#             nn.ReLU(True)
#         )

#         self.out = nn.Sequential(
#             conv( n_feats*3, n_feats, 1 ),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         # x1 = self.DDB1( x)                  # C * W * H
#         # y1 = torch.cat( [ x1, x], dim=1)    # 2C
#         # x2 = self.DDB2( y1)
#         # y2 = torch.cat( [ x2, y1], dim=1)   # 4C
#         # x3 = self.DDB3( y2)
#         # y3 = torch.cat( [ x3, y2], dim=1)   # 8C

#         # return self.out(y3)

#         # x1 = self.DDB1( x)                  # C * W * H
#         y1 = torch.cat( [ self.DDB1( x), x], dim=1)    # 2C
#         y2 = torch.cat( [ self.DDB2( y1), y1], dim=1)   # 4C
#         y3 = torch.cat( [ self.DDB3( y2), y2], dim=1)   # 8C
#         return self.out(y3)

##############################################################