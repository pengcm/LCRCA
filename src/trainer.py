import os
import math
from decimal import Decimal
import random

import utility

import torch
import torch.nn.utils as utils 
import torch.nn.functional as F

from tqdm import tqdm

# from thop import profile        # 计算 Flops & params   https://github.com/Lyken17/pytorch-OpCounter
# 關閉所有關於 self.loss. 的接口, utility.save中也有調用

class Trainer():
    ''' 网络训练 '''
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.psnr))

        # self.error_last = 1e8     # 用于和当前 loss 比较，如果当前 batch 的 loss 过大，跳过

    ##########################################################################
    ###########################################################################

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()            # learning_rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        self.model.train()

        # total, training = self.get_parameter_number(self.model)
        # self.ckp.write_log(
        #     'Total params: {}\tTraining params: {}'.format(total, training)
        # )                                   # 检验一下是否网络参数总量不变，每次训练参数量变化


        timer_data, timer_model = utility.timer(), utility.timer()
        self.loader_train.dataset.set_scale(0)

        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            # print(lr.size())
            # lr, hr = self.downsample(lr, hr, self.scale[0])
            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model( lr, 0)         # 0 是給ckp的參數
            # sr = sr + F.interpolate(lr,None,self.scale[0],'bicubic',True)
            # loss = self.loss(sr, hr,)
            loss = self.loss(sr, hr, )      

            # loss = utility.Diff_MSE(sr, hr)
            # loss2 = utility.Diff_MSE_Y(sr, hr, self.scale[0])
            # loss2 = utility.Diff_L1(sr, hr)
            

        # 训练方式选择
            # if loss.item() < self.args.skip_threshold * self.error_last:
            #     loss.backward()
            #     self.optimizer.step()
            # else:
            #     print('Skip batch {}! (Loss: {})'.format(batch + 1, loss.item()) )

            loss.backward()
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(
            #         self.model.parameters(),
            #         self.args.gclip
            #     )                   # 如果 裁切梯度，去除小於閾值的梯度，不使用它BP
            self.optimizer.step()

            timer_model.hold()
        # 日志， 训练进度
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    '[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    # '[{}/{}]\tloss:{:.4f}\tloss2:{:.4f}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,         #
                        len(self.loader_train.dataset),             # 调用 srdata.py 中 __len__
                        self.loss.display_loss(batch),
                        # loss,
                        # loss2,
                        timer_model.release(),
                        timer_data.release()    )
                )
            ## 每个 epoch 中送入 patch 数量： batch_size * train_every          ----srdata
            ## dataloader 包装 batch_size 个为一个 batch， 共计 train_every 个batch
            ## 每 print_every 记录一次，    
            timer_data.tic()

        self.loss.end_log( len(self.loader_train))
        # self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    ###########################################################################
    ###########################################################################


    def test(self):
        '''   '''
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.model.eval()           ############


        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_psnr( torch.zeros(1, len(self.loader_test), len(self.scale)) )
       
        if self.args.test_only:         # 添加 SSIM 空白日志
            self.ckp.add_ssim( torch.zeros(1, len(self.loader_test), len(self.scale)) )
        
        self.ckp.write_log( 'parameters of net: {}'.format( 
                self.get_parameter_number(self.model, False)[0])
        )               # 计算测试时的网络参数量            

        # flops, params = self.compute_flops(self.model )
        # self.ckp.write_log(
        #     'Flops of computing 720p: {}\tParams of net: {}'.format(
        #         flops, params )
        # )                           # 计算 Flops, HR 为720p, 记得修改输入参数 scale



        timer_test = utility.timer()
        # if self.args.save_results: 
        #     self.ckp.begin_background()
        
        for idx_data, d in enumerate(self.loader_test):
            eval_psnr, eval_ssim = 0.0, 0.0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=70):
                    lr, hr = self.prepare(lr, hr)
                    # sr = self.model(lr, idx_scale)              #########改改改改
                    sr = self.model( lr, idx_scale)
                    # sr = sr + F.interpolate(lr,None,scale,'bicubic',True)

                    # print('\n',sr[0,0,0,0])
                    sr = utility.quantize(sr, self.args.rgb_range)
                    # print(sr[0,0,0,0])

                ## 保存图像        
                    if self.args.save_results:      
                        save_list = [sr,lr,hr] if self.args.save_gt else [sr]       # 需保存类型
                        self.ckp.save_results(d, filename[0], save_list, scale)
                ## 写日志
                    # print(utility.Diff_MSE( 
                    #     F.interpolate(hr, None, 1 / scale, 'bicubic', True),
                    #     lr, )                 # 此处bicubic下采样后不同
                    # )                                         # 测试 SR 的下采样是否与 LR 完全一致
                    # print(utility.Diff_L1(sr, hr,),'\n'  ) 

                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )           # 计算 PSNR 总和

                    if self.args.test_only:
                        eval_ssim += utility.calc_ssim(
                        sr, hr, scale, self.args.rgb_range, 
                        )      # 计算 SSIM 总和

            ## PSNR 日志
                self.ckp.psnr[-1, idx_data, idx_scale] = eval_psnr / len(d)
                best = self.ckp.psnr.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.psnr[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                        )
                )           # 显示平均 PSNR, 各 epoch 中的最好结果，及对应 epoch， 并写入日志
            ## SSIM 日志
                if self.args.test_only:
                    self.ckp.ssim[ -1, idx_data, idx_scale] = eval_ssim / len(d)
                    self.ckp.write_log(
                        '[{} x{}]\tSSIM: {:.6f}'.format(
                           d.dataset.name,  scale, 
                           self.ckp.ssim[ -1, idx_data, idx_scale]
                        )
                    )       # 显示平均 SSIM ,并写入日志


        ## 测试耗时
        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        # self.ckp.write_log('Saving...')

        # if self.args.save_results:
        #     self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))            #####

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )       ## test 总耗时

        torch.set_grad_enabled(True)


    #################

    def downsample(self, lr, hr, scale, ):
        tar = random.random()
        if tar < 0.25:     res_scale = 2
        elif tar < 0.5:    res_scale = 3
        elif tar < 0.75:   res_scale = 4
        else:   return lr,hr

        _,_,h,w = lr.size()     ## h = w = 96
        lh = h // res_scale
        iy = random.randrange(0, h-lh+1)
        hh, ty = scale * lh, scale * iy 

        lr = lr[:, :, iy:iy + lh, iy:iy + lh].contiguous()
        hr = hr[:, :, ty:ty + hh, ty:ty + hh].contiguous()
        return  lr,hr


    ##############    把数据转化为需求格式，并导入 gpu

    def prepare(self, *args):       
        device = torch.device('cpu' if self.args.cpu else 'cuda')       # 转入gpu
        def _prepare(tensor):
            if self.args.precision == 'half': 
                tensor = tensor.half()          # 如果需求 半精度  
            return tensor.to(device,non_blocking=True)
            # return tensor.to(device, non_blocking=True)

        return [_prepare(a) for a in args]

    #############     判断 是否结束

    def terminate(self):             
        if self.args.test_only:     # 测试，一次就完成
            self.test()
            return True
        else:                       # 训练，看是否达到需求 epoch
            return self.optimizer.get_last_epoch() >= self.args.epochs


    #############    网络参数数量

    def get_parameter_number(self, net, train = True):
        total_num = sum(p.numel() for p in net.parameters())
        if train:
            train_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        else:
            train_num = 0
        return [ total_num, train_num]


    #############   计算 720p 图像的计算复杂度

    def compute_flops(self, net, scale=2):
        ''' some troubles here！！！！！！！！！！！！！！'''
        # width  = int( 1280 / scale)        # 720p 对应的 LR 尺寸
        # height = int( 720 / scale)
        test_lr = torch.rand(1, 3, 640, 360)
        test_lr = self.prepare( test_lr)      
        return profile(net, inputs=(test_lr, scale))
        # flops, params = profile(net, inputs=(input,))
        # print('Flops of 720p:{}\tparams of net: {}'.format(flops, params))


