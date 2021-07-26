import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model            # 全局变量
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():    # only_test 在判断 terminate 中
                t.train()
                t.test()            # 训练中的验证

            checkpoint.done()

if __name__ == '__main__':
    main()
