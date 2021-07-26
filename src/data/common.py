import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ''' args： lr, hr
        自原图arg中，随机取 patch_size 尺寸的一块 '''
    ih, iw = args[0].shape[:2]      # args[0], LR.  [1] HR

    if not input_large:
        p = scale if multi else 1
        hp = patch_size *p        # high_patch
        lp = hp // scale            # low_patch
    else:
        hp = patch_size
        lp = patch_size

    ix = random.randrange(0, iw - lp + 1)       # 宽度上，随机取点开始，LR
    iy = random.randrange(0, ih - lp + 1)       # 高度

    if not input_large:
        tx, ty = scale * ix, scale * iy         # HR
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + lp, ix:ix + lp, :],
        *[a[ty:ty + hp, tx:tx + hp, :] for a in args[1:]]
    ]

    return ret



def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]



def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]



def augment(*args, ):           # 旋转/翻转 扩充数据集
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    # hflip, vflip, rot90 = 0,0,1

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        # img.transpose(1, 0, 2)
        return img

    return [_augment(a) for a in args]
