import argparse
import template

parser = argparse.ArgumentParser(description=' TEST SR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='only use cpu in dataloader')        # 在 dataloader 中只使用cpu，不拷贝到cuda
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=11,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../DataSet',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../../DataSet/benchmark',
                    help='demo image directory')                # 测试图像位置
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')                  # 可 + 
parser.add_argument('--data_test', type=str, default='Set14',
                    help='test dataset name')                   # Set5+Set14+B100+Urban100+DIV2K
parser.add_argument('--data_range', type=str, default='1-800/881-900',      # 训练 / 验证集
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')        # 建立新文件夹保存 .pt  benchmark就无 .pt
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')        # 是否通过旋转来拓展数据集，默认有扩充    

parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')       # 16
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')                 # lr size * scale = patch size  
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# Model specifications
parser.add_argument('--model', default='SRCN',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')                 # relu方式
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')   # 16
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')

parser.add_argument('--res_scale', type=float, default=1.0,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')                 # 采用膨胀卷积
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')   # 数据精度控制

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)s
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--train_every', type=int, default=1000,
                    help='how many loss backwards per epoch')   # 单次 epoch 更新权重次数--srdata
parser.add_argument('--epochs', type=int, default=700,
                    help='number of epochs to train')           # epoch = 300  提升 0.03 左右
parser.add_argument('--decay', type=str, default='400-500-600',
                    help='learning rate decay type: 100-200-300')    # lr 减小于第 xx 个 epoch
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')                       # default = e-4  SGD:-7
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')    # lr 减小倍率
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')

# Testing specifications
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')   ###
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')   # TEST

# Optimization specifications
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')             # L1,MSE

parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')     # 可在训练时使用

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')            # 保存每个epoch的参数
parser.add_argument('--print_every', type=int, default=200,
                    help='how many batches to wait before logging training status')  # 多久记录                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     一次
parser.add_argument('--save_results', action='store_true',
                    help='save output images while testing')        # 测试时，保存测试结果
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together while testing')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e2

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

