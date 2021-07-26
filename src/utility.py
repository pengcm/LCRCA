import os
import math
import time
import datetime
# from multiprocessing import Process
# from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import cv2


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.psnr = torch.Tensor()           # psnr
        self.ssim = torch.Tensor()


        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)      # 保存位置，训练时使用
        else:
            self.dir = os.path.join('..', 'experiment', args.load)      # 载入文件夹位置，继续训练     
            if os.path.exists(self.dir):
                self.psnr = torch.load(self.add_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.psnr)))
            else:
                args.load = ''

        if args.reset:                                  # reset 时，日志文件夹删除重建
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)            # 创建文件夹
        os.makedirs(self.add_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.add_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.add_path('log.txt'))else 'w'
        self.log_file = open(self.add_path('log.txt'), open_type)
        with open(self.add_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def add_path(self, *subdir):                       # 在当前路径 dir 中加入新的文件夹
        return os.path.join(self.dir, *subdir)      

    def save(self, trainer, epoch, is_best=False):     # 训练，保存一系列文件
        trainer.model.save(self.add_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.psnr, self.add_path('psnr_log.pt'))

    def add_psnr(self, log):                # 新建空间存储 psnr
        self.psnr = torch.cat([self.psnr, log])
    
    def add_ssim(self, log):
        self.ssim = torch.cat([self.ssim, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.add_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.psnr[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.add_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    # def begin_background(self):         # 测试，起点标记 ？？
    #     self.queue = Queue()

    #     def bg_target(queue):
    #         while True:
    #             if not queue.empty():
    #                 filename, tensor = queue.get()
    #                 if filename is None: break
    #                 imageio.imwrite(filename, tensor.numpy())
        
    #     self.process = [
    #         Process(target=bg_target, args=(self.queue,)) \
    #         for _ in range(self.n_processes)
    #     ]
        
    #     for p in self.process: p.start()

    # def end_background(self):           # 测试，结束点 ？？？
    #     for _ in range(self.n_processes): self.queue.put((None, None))
    #     while not self.queue.empty(): time.sleep(1)
    #     for p in self.process: p.join()

    # def save_results(self, dataset, filename, save_list, scale):    # 保存测试结果图像
    #     # if self.args.save_results:
    #     filename = self.add_path(
    #         'results-{}'.format(dataset.dataset.name),
    #         '{}_x{}_'.format(filename, scale)
    #     )                               # 待写入文件的位置   eg.  ../../woman_x2_SR.png

    #     postfix = ('SR', 'LR', 'HR')
    #     for v, p in zip(save_list, postfix):                            # p 是后缀 SR
    #         normalized = v[0].mul(255 / self.args.rgb_range)            # 归一化 [0,1]
    #         tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()       # gpu文件转入cpu
    #         self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))    # 写入图像

    def save_results(self, dataset, filename, save_list, scale):    
        filename = '{}/results-{}/{}_x{}_'.format(
            self.dir, dataset.dataset.name, 
            filename, scale
        )
        # if not os.path.exists(filename):
        #     os.makedirs(filename)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format(filename, p), ndarr)




def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

##########################

def calc_psnr(sr, hr, scale, rgb_range, dataset=None,):          ### PSNR
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / (rgb_range + 1)
        diff = diff.mul(convert).sum(dim=1)             # Y-channel

    # scale = dist_side * scale
    diff = diff[..., scale:-scale, scale:-scale]
    mse = diff.pow(2).mean() 
    psnr = -10 * math.log10(mse) if mse > 2.0e-6 else 60

    return psnr


######### BasicSR -xintao wang##################
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(sr, hr, scale, rgb_range,):
    '''calculate SSIM, the same outputs as MATLAB's
    sr , hr [0, 255]  N*C*H*W = 1*3*H*W     （
    img1, img2: [0, 255]
    '''
    if not sr.shape == hr.shape:
        raise ValueError('Input images must have the same dimensions.')

    if sr.size(1) > 1:
        convert = sr.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        sr = sr.mul_(convert).div_(256).sum(dim=1, keepdim=True)
        hr = hr.mul_(convert).div_(256).sum(dim=1, keepdim=True)

    sr = sr[0,0, scale:-scale, scale:-scale].to('cpu').numpy()
    hr = hr[0,0, scale:-scale, scale:-scale].to('cpu').numpy()

    return ssim(sr, hr)


######################################################
def Mean_Var(tensors):
    '''计算 （N * C） * H * W 后两维的均值、方差
    输入 torch.Tensor '''
    miu = torch.mean( tensors, (-2,-1))
    sigma = torch.var( tensors, (-2,-1))

    return miu, sigma
 

def Diff_L1(tensor1, tensor2):
    ''' 主要用于 （N * C） 大小的均值和方差矩阵，计算两者的差异程度，用 L1 '''
    if tensor1.size() != tensor2.size():
        raise ValueError ('Input features must have the same dim!-sp')
    diff = tensor1 - tensor2
    loss = diff.norm(1) / torch.numel(diff)         # 1 范数 求均值 --> L1
    return loss

def Diff_MSE(tensor1, tensor2):
    if tensor1.size() != tensor2.size():
        raise ValueError ('Input features must have the same dim!-sp')
    diff = tensor1 - tensor2
    # loss = diff.norm(2).pow(2) / torch.numel(diff)         #
    loss = diff.pow(2).mean()        #
    return loss

def Diff_MSE_Y(tensor1, tensor2, scale):
    ''' 和計算PSNR 一樣的計算方式, Y通道修邊後計算'''
    diff = tensor1 - tensor2
    gray_coeffs = [65.738, 129.057, 25.064]
    convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    diff = diff.mul(convert).sum(dim=1)     # Y-channel

    diff = diff[..., scale:-scale, scale:-scale]        # shave the edge
    loss = diff.pow(2).mean()       #
    return loss

####################################################### 
def Local_Var(imgs, kernel_size=7):
    ''' 计算局部方差, 输入 图像，局部尺寸'''
    n,c,h,w = imgs.size()
    if h<kernel_size or w<kernel_size:
        raise ValueError(" conv kernel can not bigger than input image! -sp")
    out = torch.zeros( imgs.size()).cuda()
    flip = kernel_size - 1
    for ni in range(n):
        for ci in range(c):
            for hi in range(h-flip):
                for wi in range(w-flip):
                    conv = imgs[ni,ci, hi:hi+flip, wi: wi+flip]
                    out[ni,ci,hi,wi]= torch.var( conv, (0,1))
    return out

def Local_Loss(sr_var, hr_var):
    ''' 计算 fai( hr_var)*(hr_var - sr_var) '''
    if sr_var.size() != hr_var.size():
        raise ValueError ('Input sigmas must have the same dim!-sp')
    diff = torch.abs( hr_var - sr_var)
    diff = hr_var * diff          # fai(x) = 1*x
    loss = diff.norm(1) / torch.numel(diff)
    return loss
#######################################################


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler

    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR
    # scheduler_class = lrs.StepLR()

    # kwargs_scheduler = {'T_0': 50, 'eta_min': 0.00001}
    # scheduler_class = lrs.CosineAnnealingWarmRestarts
    # kwargs_scheduler = {'base_lr':0.00001, 'max_lr':0.0002, 'step_size_up':50, 'mode':'triangular2' }
    # scheduler_class = lrs.CyclicLR


    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.add_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.add_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def add_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

