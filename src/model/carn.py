## Shu
## 2019-7
### UESTC kb249
##==================================================================================
import torch
import torch.nn as nn
import model.ops as ops


def make_model(args, parent=False):
    return CARN(args)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        #b1 = self.b1(x)
        x = torch.cat([x, self.b1(x)], dim=1)
       # o1 = self.c1(c1)

        # b2 = self.b1(self.c1(c1))
        x = torch.cat([x,  self.b1(self.c1(x))], dim=1)
       # o2 = self.c2(c2)
        
        #b3 = self.b1(self.c2(c2))
        #c3 = torch.cat([c2, self.b1(self.c2(c2))], dim=1)
       # o3 = self.c3(c3)

        return  self.c3(torch.cat([x, self.b1(self.c2(x))], dim=1))



class CARN(nn.Module):
    def __init__(self, args, multi_scale=0, group=1):
        super(CARN, self).__init__()
        
        self.scale = args.scale[0]
       # multi_scale = args.get("multi_scale")
       # group = args.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64, group=group)
        self.b2 = Block(64, 64, group=group)
        self.b3 = Block(64, 64, group=group)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = ops.UpsampleBlock(64, scale=self.scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        # c0 = o0 = x

       # b1 = self.b1(x)
        x = torch.cat([x, self.b1(x)], dim=1)
        # o1 = self.c1(c1)
        
       # b2 = self.b2( self.c1(c1))
        c2 = torch.cat([x, self.b2( self.c1(x))], dim=1)
       # o2 = self.c2(c2)
        
       # b3 = self.b3(o2)
        out = torch.cat([c2, self.b3(self.c2(c2))], dim=1)
       # o3 = self.c3(c3)

        out = self.upsample(self.c3(out), scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out