###### sp

from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return TEST(args)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                bias = True, redu=2):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential( 
            conv(n_feats, n_feats//2,kernel_size,bias=bias),
            nn.ReLU(True),
            conv(n_feats//2, n_feats,kernel_size,bias=bias),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential( 
            conv(n_feats, n_feats//2,kernel_size,bias=bias),
            nn.ReLU(True),
            conv(n_feats//2, n_feats,kernel_size,bias=bias),
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv2( x + x1)
        return x + x1


class ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias):
        super(ResGroup, self).__init__()

        self.b1 = ResBlock(conv, n_feats, kernel_size, bias)
        self.b2 = ResBlock(conv, n_feats, kernel_size, bias)
        self.b3 = ResBlock(conv, n_feats, kernel_size, bias)
        # self.cat1 = conv( 2 * n_feats, n_feats, 1, bias)
        # self.cat2 = conv( 3 * n_feats, n_feats, 1, bias)
        #  

        self.sca =  common.SCALayer(n_feats, bias)
        # self.conv11 = conv(2*n_feats, n_feats, 1)


    def forward(self, x):
        x1 = self.b1(x)
        # cat1 = torch.cat([x,x1],dim=1)
        # x2 = self.b2( self.cat1( cat1))
        # x3 = self.b3( self.cat2( torch.cat([cat1, x2], dim=1)))
        # x = x1 + x2 + x3

        x2 = self.b2(x1)
        x3 = self.b3(x2)
        # x = self.cat2(torch.cat([x1,x2,x3],dim=1))

        y = self.sca(x3)
        # x = self.conv11( torch.cat([x,y], dim=1))
        return y


class TEST(nn.Module):
    def __init__(self, args, bias=True,
                conv = common.default_conv):
        super(TEST, self).__init__()

        n_feats = 64
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size, bias)

        # define body module
        self.bodypre = conv(n_feats,n_feats,kernel_size,bias)
        self.b1 = ResGroup(conv, n_feats, kernel_size, bias)
        self.b2 = ResGroup(conv, n_feats, kernel_size, bias)
        self.b3 = ResGroup(conv, n_feats, kernel_size,bias)
        self.cat1 = conv( 2 * n_feats, n_feats, 1, bias)
        self.cat2 = conv( 3 * n_feats, n_feats, 1, bias)
        # self.cat3 = conv( 4 * n_feats, n_feats, 1, bias)

        # define tail module
        self.tail = common.UpInter(conv, scale, n_feats, args.n_colors, bias=bias)
        # self.mainpart = F.interpolate

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        # res = self.body(x)
        # x = x + res

        xpre = self.bodypre(x)
        x1 = self.b1(xpre)
        # x2 = self.b2(x1)
        # x3 = self.b2(x2)

        cat1 = torch.cat([xpre,x1],dim=1)
        x2 = self.b2( self.cat1( cat1))

        cat2 = torch.cat([cat1, x2], dim=1)
        x3 = self.b3( self.cat2( cat2))
        x = x + x1 + x2 + x3
        # x = self.cat3( torch.cat([x,x1,x2,x3],dim=1))
        # x = x + x3
        # x = x + res

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