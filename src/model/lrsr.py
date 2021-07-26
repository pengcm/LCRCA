###### sp

from model import common

import torch
import torch.nn as nn
import torch.nn.init as init

def make_model(args, parent=False):
    net = LRSR(args)
    # classname=net.__class__.__name__
    # if classname.find('Conv') != -1:
    #     init.kaiming_uniform_(net.weight.data,a=0,nonlinearity='relu')
    #     init.kaiming_uniform_(net.bias.data,a=0,nonlinearity='relu')
    #     # kaiming_normal_
    return net


class LRSR(nn.Module):
    def __init__(self, args, bias=True,
                conv = common.default_conv):
        super(LRSR, self).__init__()

        n_feats = 32
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size,bias)

        # define body module
        self.bodypre = conv(n_feats,n_feats,kernel_size,bias)
        self.b1 = common.ResGroup(conv, n_feats, kernel_size, bias)
        self.b2 = common.ResGroup(conv, n_feats, kernel_size, bias)
        self.b3 = common.ResGroup(conv, n_feats, kernel_size, bias)
        self.cat1 = conv( 2 * n_feats, n_feats, 1, bias)
        self.cat2 = conv( 3 * n_feats, n_feats, 1, bias)

        # define tail module
        self.tail = common.UpInter(conv, scale, n_feats, args.n_colors, bias=bias)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        xpre = self.bodypre(x)
        # x = x + xpre
        x1 = self.b1(xpre)
        # x = x + x1

        cat1 = torch.cat([xpre,x1],dim=1)
        x2 = self.b2( self.cat1( cat1))
        # x = x + x2

        cat2 = torch.cat([cat1, x2], dim=1)
        x3 = self.b3( self.cat2( cat2))
        x = x + x1 + x2 + x3

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