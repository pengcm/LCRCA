###### sp

from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return SRCNMORE(args)


''' 
对比试验：
        midBlock --普通残差块
        DenseResBlock --串联结构
'''
class MidBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                bias = True, res_scale=1.0, redu=2):
        super(MidBlock, self).__init__()

        # self.conv1 = nn.Sequential( 
        #     conv(n_feats, n_feats,kernel_size,bias=bias),
        #     nn.ReLU(True),
        #     conv(n_feats, n_feats,kernel_size,bias=bias),
        # )
        self.conv1 = conv(n_feats,n_feats//redu, kernel_size,bias=bias)
        self.conv2 = nn.ModuleList( 
                nn.Sequential(
                conv(n_feats//redu,n_feats//redu, kernel_size,bias=bias),
                nn.ReLU(True),
                conv(n_feats//redu,n_feats//redu, kernel_size,bias=bias),
            ) for _ in range(2)
        )
        self.conv4 = conv(n_feats//redu,n_feats, kernel_size,bias=bias)

            
    def forward(self,x):
        # x1 = self.conv1(x)
        # return x + x1
        x1 = self.conv1(x)
        x1 = self.conv2[0](x1) + x1
        x1 = self.conv2[1](x1) + x1
        x  = self.conv4(x1) + x
        return x
        

class DenseResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, ):
        super(DenseResBlock, self).__init__()

        self.b1 = MidBlock(conv, n_feats, kernel_size)
        self.b2 = MidBlock(conv, n_feats, kernel_size)
        self.b3 = MidBlock(conv, n_feats, kernel_size)
        # self.cca = common.SCALayer(n_feats)

    def forward(self, x):
        x = self.b1( x)
        x = self.b2( x)
        x = self.b3( x)
        # x = self.cca(x)
        return x


class SRCNMORE(nn.Module):
    def __init__(self, args,
                conv = common.default_conv):
        super(SRCNMORE, self).__init__()

        n_feats = 64
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size)

        # define body module
        self.bodypre = conv(n_feats,n_feats,kernel_size)
        self.b1 = DenseResBlock(conv, n_feats, kernel_size)
        self.b2 = DenseResBlock(conv, n_feats, kernel_size)
        self.b3 = DenseResBlock(conv, n_feats, kernel_size)

        # define tail module
        self.tail = common.UpInter(conv, scale, n_feats, args.n_colors, )
        

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        xpre = self.bodypre(x)
        x1 = self.b1(xpre)
        x1 = self.b2( x1)
        x1 = self.b3( x1)
        x = x + x1

        x = self.tail(x)
        x = self.add_mean(x)

        return x

    

    def load_state_dict(self, state_dict, strict=False):
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        if name.find('tail') >= 0:
                            print('Replace pre-trained upsampler to new one...')
                        else:
                            raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
                elif strict:
                    if name.find('tail') == -1:
                        raise KeyError('unexpected key "{}" in state_dict'
                                    .format(name))

            if strict:
                missing = set(own_state.keys()) - set(state_dict.keys())
                if len(missing) > 0:
                    raise KeyError('missing keys in state_dict: "{}"'.format(missing))