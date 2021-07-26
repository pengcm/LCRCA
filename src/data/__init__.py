from importlib import import_module
# from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

## 训练取原图中 patch_size 大小的块。 
## 测试，LR不变， HR 裁切至 scale * shape(LR)

## 验证，validation 控制是否对测试图像取块
## 在 srdata 中修改 repeat 来调整取块数量

## data  <--  div2kjepg / benchmark   <--   srdata   <--   common 


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:           #'--data_train', type=str, default='DIV2K'
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())            # data 中的 div2kjpeg.py
                datasets.append(getattr(m, module_name)(args, validation=False, name=d))      
                    # 调用 .py 中的 DIV2KJPEG 类
                    # 調用 .py中對應的class，繼承自 srdata，調用common.get_patch，確定 patchsize

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,      # copy data into CUDA pinned memory before returning them
                num_workers=args.n_threads,    # 子线程数
            )                                  # 原作者更新版本

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:      # 测试集
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, name=d) # 调用对应文件中的 class

            else:                                               # 验证集取 DIV2K 801--900
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(
                    args, train=False, validation=False, name=d) # 验证 
                    # args, train=False, validation=not args.test_only, nme=d) # 验证 
                    # 使用 DIV2K 验证，valid = True，取块作测试。 # 作测试集，valid = False, 整张图作测试

            self.loader_test.append(  dataloader.DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            ))                          # 原作者更新版本

