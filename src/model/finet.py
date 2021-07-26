from model import common
import torch
import torch.nn as nn

######### python main.py --model FINET --scale 2 --n_resblocks 4 --patch_size 96 --save XXXXXXXX --reset

def make_model(args, parent=False):
    return FINET(args)
     


class FINET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FINET, self).__init__()

        n_resblocks = args.n_resblocks          # 深度可分卷积密集连接模块的数量
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = nn.Sequential(
            conv(args.n_colors, n_feats, kernel_size),
        )       ## batch_size * n_feats * H *W

        # define body module
        # m_body = [ common.DepDenseBlock() for _ in range(n_resblocks) ]
        # m_body.append( conv(n_feats, n_feats, kernel_size))

        # self.body = nn.Sequential(*m_body)

        self.body1 = common.DepDenseBlock(n_feats)
        self.body2 = common.DepDenseBlock(n_feats)
        self.body3 = common.DepDenseBlock(n_feats)
        self.body4 = common.DepDenseBlock(n_feats)
        self.conv11 = conv(2*n_feats, n_feats, 1)
        self.conv12 = conv(3*n_feats, n_feats, 1)
        self.conv13 = conv(4*n_feats, n_feats, 1)
        self.conv14 = conv(5*n_feats, n_feats, 1)
        self.conv1  = conv(  n_feats, n_feats, 3)

        # self.bodyconv = nn.Sequential(
        #     conv(4*n_feats, n_feats, 1),
        #     conv(n_feats,n_feats,3)
        #     )


        # define tail module
        self.tail = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        )


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        # x = self.body(x) + x
        y1 = self.body1(x)
        c1 = torch.cat([x,y1],dim=1)
        x2 = self.conv11(c1)

        y2 = self.body2(x2)
        c2 = torch.cat([c1,y2],dim=1)
        x3 = self.conv12(c2)

        y3 = self.body3(x3)
        c3 = torch.cat([c2,y3],dim=1)
        x4 = self.conv13(c3)

        y4 = self.body4(x4)
        c4 = torch.cat([c3,y4],dim=1)
        res = self.conv14(c4)
        x = x + self.conv1( res)

        x = self.tail(x)
        x = self.add_mean(x)

        return x 

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

