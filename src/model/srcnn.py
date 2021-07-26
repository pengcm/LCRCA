###### sp

from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return SRCNN(args)

class SRCNN(nn.Module):
    def __init__(self, args,
                conv = common.default_conv):
        super(SRCNN, self).__init__()

    #     self.scale = args.scale[0]
    #     self.sub_mean = common.MeanShift(args.rgb_range)
    #     self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    #     self.layers = torch.nn.Sequential(
    #         nn.Conv2d( args.n_colors, 16,   3, 1, 1),
    #         nn.ReLU(True),

    #         nn.Conv2d( 16, 16, 3, 1, 1),
    #         nn.ReLU(True),

    #         nn.Conv2d( 16, 16, 3, 1, 1),
    #         nn.ReLU(True),

    #         nn.Conv2d( 16, 16, 3, 1, 1),
    #         nn.ReLU(True),
    #     )
    #     # self.final_net =nn.Sequential( 
    #     #     common.Upsampler(conv, self.scale, 16, act=False),
    #     #     conv( 16, , 3),
    #     # )     
    #     self.final_net = nn.Conv2d(16, args.n_colors, 3, 1, 1)

    # def forward(self, x):
    #     # x = nn.functional.interpolate(x, None, self.scale, 'bicubic', False)
    #     x = self.sub_mean(x)
    #     x = self.layers(x)
    #     x = self.final_net(x)
    #     x = self.add_mean(x)
    #     return x

        self.n_resblocks = 8
        n_feats = 64
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     common.ResBlock(
        #         conv, n_feats, kernel_size, act=True, res_scale=args.res_scale
        #     ) for _ in range(n_resblocks)
        # ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.ModuleList(
            common.MidBlock(conv, n_feats, kernel_size,) for _ in range (self.n_resblocks)
        )
        self.body2 = conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        # x = self.body(x) + x
        x1 = self.body[0](x)
        res = x1

        for n in range (self.n_resblocks-1):
            x1 = self.body[n+1](x1)
            res = res + x1

        res = self.body2(res)
        x = x + 0.1 *res

        x = self.tail(x)
        x = self.add_mean(x)

        return x

        # x2 = self.body[1](x1)
        # res = res + x2
        # x3 = self.body[2](x2)
        # res = res + x3
        # x4 = self.body[3](x3)
        # res = res + x4
        # x5 = self.body[4](x4)
        # x6 = self.body[5](x5)
        # x7 = self.body[6](x6)
        # x8 = self.body[7](x7)
        # x9 = self.body[8](x8)
        # x10 = self.body[9](x9)
        # x11 = self.body[10](x10)
        # x12 = self.body[11](x11)

       
        # x9 = x + 0.1* (x1 +x2 +x3 +x4 +x5 +x6 +x7 +x8)
        # x = x + 0.1*(x1 +x2 +x3 +x4 +x5 +x6 +x7 +x8)



    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
    

