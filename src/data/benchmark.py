import os

from data import srdata
# from data import common
# import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=False, validation=False, benchmark=True):
        # 参数传入 srdata.SRData
        super(Benchmark, self).__init__(
            args, name=name, train=train, validation=validation, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')




