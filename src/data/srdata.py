import os
import glob
import random
import pickle

from data import common

# import numpy as np
import imageio
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, validation=True, benchmark=False):
        # 在__init__中，设置是 训练 or 验证 模式, 两者都 False 时，测试模式
        self.args = args
        self.name = name
        self.train = train
        self.validation = validation
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')       # 当输入尺寸大小等同输出时
        self.scale = args.scale
        self.idx_scale = 0
        self.repeat = 1
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            # 将 .png 替换为 .pt
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.train_every           # 一个epoch中的总块数
            # n_patches = args.train_every 
            n_images = len(args.data_train) * len(self.images_hr)   # 1 * 800
            self.repeat = max(n_patches // n_images, 1)             # div2k 中训练集的循环次数
                # 因为 b_s 一般为 8/16，因此 n_patches 会是 800 的倍数，
        elif validation:
            self.repeat = 20         # 手动设置验证集重复次数       #########################


    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))

        return names_hr, names_lr


    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train or self.validation:
            return len(self.images_hr) * self.repeat        # 800/100   * 重复次数
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train or self.validation:
            return idx % len(self.images_hr)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename


    def get_patch(self, lr, hr):
        ''' 1 patch once'''
        scale = self.scale[self.idx_scale]
        if self.train or self.validation:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment and self.train:
                lr, hr = common.augment(lr, hr)   

        else:   # 测试集
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr


    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

