###### sp

from model import common

import torch
import torch.nn as nn
import torch.nn.init as init

def make_model(args, parent=False):
    net = GRRN(args)
    classname=net.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_uniform_(net.weight.data,a=0,nonlinearity='relu')
        init.kaiming_uniform_(net.bias.data,a=0,nonlinearity='relu')
        print('initialing the net')
        # kaiming_normal_  kaiming_uniform_
    return net


class ReBlock(nn.Module):
    def __init__(self, n_feats, kernel_size,
                bias = True, ):
        super(ReBlock, self).__init__()

        self.conv1 = nn.Sequential( 
            nn.Conv2d(n_feats, n_feats//2, (3,3), 1, (1,1),bias=bias),
            nn.ReLU(True),
            nn.Conv2d(n_feats//2, n_feats, (3,3), 1, (1,1),bias=bias),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential( 
            nn.Conv2d(n_feats, n_feats//2, (3,3), 1, (1,1),bias=bias),
            nn.ReLU(True),
            nn.Conv2d(n_feats//2, n_feats, (3,3), 1, (1,1),bias=bias),
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv2(x +x1)
        return x + x1

class ResGroup(nn.Module):
    def __init__(self, n_feats, kernel_size, bias):
        super(ResGroup, self).__init__()

        self.b1 = ReBlock( n_feats, kernel_size, bias)
        self.b2 = ReBlock( n_feats, kernel_size, bias)
        self.b3 = ReBlock( n_feats, kernel_size, bias)
        self.relu = nn.ReLU(True)
        self.sca =  common.SCALayer(n_feats, bias)

    def forward(self, x):
        x1 = self.relu( self.b1(x))
        x2 = self.relu( self.b2(x1))
        x3 = self.b3(x2) 
        out = self.sca(x3)

        return out 


class GRRN(nn.Module):
    def __init__(self, args, bias=True,
                conv = common.default_conv):
        super(GRRN, self).__init__()

        n_feats = 64
        kernel_size = 3 

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = conv(args.n_colors, n_feats, kernel_size, bias)

        # define body module
        self.bodypre = conv(n_feats,n_feats,kernel_size,bias)
        self.group = nn.ModuleList(
            ResGroup( n_feats, kernel_size, bias) for _ in range(3)
        )
        # self.concat = nn.ModuleList(
        #     nn.Conv2d(2* n_feats, n_feats, 1, 1, 0,bias=bias) for _ in range(3)
        # )

        # define tail module
        self.tail = common.UpInter(conv, args.scale[0], n_feats, args.n_colors, bias=bias)
        

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        xpre = self.bodypre(x)
        x1 = self.group[0](xpre) 
        x2 = self.group[1](x1) 
        res = self.group[2](x2) 

        # cat2 = torch.cat([xpre,x1],dim=1)
        # x2 = self.group[1]( self.concat[0]( cat2))

        # cat3 = torch.cat([x1,x2],dim=1)
        # x3 = self.group[2]( self.concat[1]( cat3))

        # res = self.concat[2](torch.cat([x2,x3],dim=1))

        x = self.tail(x+ res)
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