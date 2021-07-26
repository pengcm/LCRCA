import os.path
import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm


import data
import utility
from option import args
from model.srcnn import SRCNN         # example


'''

sp-kb249-2020-1-6

在训练时采用小块作为验证，此文件用于测试，将网络生成的小块拼接为大图

需要在 args 中设置的参数：
    --scale 2           # or 3 or 4
    --data_test Set5 + Set14
    --test_only
    --save_results      # help to save results of the test

'''



def main(args, loader_test, ckp):

    device = torch.device('cpu' if args.cpu else 'cuda') 
    ckp.write_log('\nTest on Benchmark:' )
    ckp.add_psnr( torch.zeros(1, len(loader_test), len(args.scale)) )

    # for idx_scale, scale in enumerate(args.scale):
    idx_scale = 0               ##一次处理 1 个 scale 
    scale = args.scale[0]

    ##
    ##-------------- 载入模型及参数 ----------------

    model = SRCNN(args)
    model = model.to(device)

    # 参数 位置
    param_path = os.path.join('../experiment/{}/model/model_best.pt'.format('srcnn_test1_15'))     
    # param_path = os.path.join('test_models', 'SRCNN'+'x%01d.pt' % (scale))        # 放在一起时

    model.load_state_dict( torch.load(param_path), strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    ckp.write_log('net params: {}'.format(
        sum( p.numel() for p in model.parameters()) )
    )


    ##
    ##--------------- 分块超分后拼接 ----------------
    lr_size = 6                # 训练时 LR.size = args.patch_size // scale
    # dist_side = 2               # padding = 无用边缘, distance to side
    stride = lr_size - 2 * args.dist_side      # 单次有效的 SR_size, > 1 可能取不完 LR 点

    timer_test = utility.timer()
    if args.save_results:  ckp.begin_background()

    for idx_data, d in enumerate(loader_test):
        ''' 多个测试集逐一 SR '''
        d.dataset.set_scale(idx_scale)

        idx_d = 0
        for lr, hr, filename in d:
            idx_d = idx_d + 1
            print(filename,'\t',idx_d,'/',len(d))
            
            lr = lr.to(device)      # [1, 3, H, W]
            hr = hr.to(device)      # [1, 3, H*scale, W*scale]

            ## 对 lr 分块处理
            lr_p = F.unfold( lr, (lr_size,lr_size), 1, args.dist_side, stride)

            # ( , kernel_size,dilation,padding,stride)      [1, 3*lr_s*lr_s, patch_num]
            # patch_num = pi{ [lr.size-2*pad-k_s]//stride +1 }  
            #   由于stride = lr.size-2*pad      ==》 pi{ lr.size//stride}

            lr_p = lr_p.transpose(1,2).transpose(0,1)         # [patch_num, 3*lr_s*lr_s]
            lr_p = lr_p.contiguous().view( -1, 1, 3, lr_size, lr_size)  # [p_n, 1, 3, lr_s, --]
        
                # sr0 = model(lr0)
                
            sr_p = torch.zeros(
                [lr_p.size(0),1, 3, lr_size*scale, lr_size*scale],
                dtype= lr_p.dtype, device= lr_p.device,
            )               # [p_n, 1, 3, lr_s*scale, --]

            for p in tqdm( range( lr_p.size(0)), ncols=70):
                sr_p[p,...] = model( lr_p[p,...])  


            shave = scale*args.dist_side      
            sr_p = sr_p.squeeze(1)                              # [p_n,1,3,... ]
            sr_p = sr_p[..., shave: -shave, shave:-shave]       # [p_n,  3, stride*scale, --]

            # ## sr_p 拼接为sr, sr 的尺寸, 修改psnr计算前，将hr。size-->
            lr_h, lr_w = lr.size(-2)//stride, lr.size(-1)//stride       # p_n = lr_h*lr_w
            K1, K2= sr_p.size(-2), sr_p.size(-1)          # k1,k2
            
            sr_p = sr_p.transpose(0,1)
            sr_p = sr_p.contiguous().view(3,lr_h,lr_w,K1,K2)
            sr_p = sr_p.transpose(2,3)
            sr_p = sr_p.contiguous().view(3,lr_h,K1,lr_w*K2)
            sr_p = sr_p.contiguous().view(3,lr_h*K1,lr_w*K2)
            sr = sr_p.unsqueeze(0)                    # [1, 3, lr_h*K1, lr_w*K2]

            sr = utility.quantize( sr, args.rgb_range)

            ## 注意计算区域
            ckp.psnr[-1, idx_data, idx_scale] += utility.calc_psnr( 
                        sr, hr, scale, args.dist_side, args.rgb_range, dataset=d)        

            if args.save_results:       # /experiment/test/...
                ckp.save_results(d, filename[0], [sr], scale)
            
        ckp.write_log( '[{} x{}]\tPSNR: {:.3f} '.format(
                    d.dataset.name,  scale, 
                    ckp.psnr[-1, idx_data, idx_scale] / len(d),   ) 
                    )  

    ckp.write_log('Forward: {:.2f}s'.format(timer_test.toc()))
    if args.save_results:  ckp.end_background()
    ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True )

    






checkpoint = utility.checkpoint(args)

if __name__ =='__main__':
    '''千万 --test_only， 一次处理 1 个 scale '''
    loader = data.Data(args)            ## 数据集
    loader_test = loader.loader_test

    main(args, loader_test, checkpoint)


