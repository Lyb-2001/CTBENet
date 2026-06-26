import os
import shutil
import json
import time
from torch.cuda import amp
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox.metrics_sus import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger, AdamW
from toolbox import setup_seed
from torch.utils.tensorboard import SummaryWriter
from toolbox.losses.lovasz_losses import lovasz_softmax
import torch.nn.functional as F
setup_seed(33)
CUDA_LAUNCH_BLOCKING=1
def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            )

class eeemodelLoss(nn.Module):

    def __init__(self, device, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        # self.class_weight_semantic = torch.from_numpy(np.array(
        #     [1.12274157,2.08751514,25.38980139,28.42099271,10.00449854,32.97445065])).float()
        self.class_weight_binary = torch.from_numpy(np.array([2.0197, 2.9765])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4584, 18.7187])).float()

        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()

        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        # self.semantic_loss = nn.CrossEntropyLoss()
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)


        # self.sdf_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
    def forward(self, inputs, targets):
        out1,out2,out3,b,_ = inputs
        semantic_gt, boundary_gt, binary_gt = targets
        # gradient_gt = gradient_gt.unsqueeze(1)
        # loss_lv1 = lovasz_softmax(F.softmax(out1,dim=1),semantic_gt)
        # loss_lv2 = lovasz_softmax(F.softmax(out2,dim=1),semantic_gt)
        # loss_lv3 = lovasz_softmax(F.softmax(out3,dim=1),semantic_gt)
        # loss_lv4 = lovasz_softmax(F.softmax(out4,dim=1),semantic_gt)

        # print(binary_gt.shape)
        # print(down_gt(binary_gt,h1.size()[2:]))
        loss1 = self.semantic_loss(out1, semantic_gt)#+0.5*loss_lv1
        loss2 = self.semantic_loss(out2,semantic_gt)
        loss3 = self.semantic_loss(out3,semantic_gt)
        # loss4 = self.cross_entropy(out4,semantic_gt)
        loss5 = self.boundary_loss(b,boundary_gt)



        loss =loss1+0.5*loss2+0.3*loss3+loss5
        return loss

def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    ### multi-gpus
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    ###
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    # model.load_state_dict(torch.load(
    #     '/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/b4/200model.pth'))
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    ## multi-gpus
    model.to(gpu_ids[0])
    model = torch.nn.DataParallel(model, gpu_ids)

    # dataloader
    trainset, valset, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True, drop_last=True)
    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    train_criterion = eeemodelLoss(gpu_ids[0]).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0
    best_miou = 0.0
    best_avg = 0.0
    best_epo = 0
    # 每个epoch迭代
    # 每个epoch迭代循环
    log_dir = os.path.join("logs","seed_1")
    writer = SummaryWriter(log_dir)
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].cuda()
                label = sample['label'].cuda()
                bound = sample['boundary'].cuda()
                binary = sample['binary'].cuda()
                targets = [label, bound, binary]
                # predict = model(image)
            else:
                image = sample['image'].cuda()
                depth = sample['depth'].cuda()
                label = sample['label'].cuda()
                bound = sample['boundary'].cuda()
                binary = sample['binary'].cuda()
                # gradient = sample['gradient'].cuda()
                targets = [label,bound,binary]
            with amp.autocast():
                # predict = model(image, depth)
                # print(image.shape)
                predict = model(image,depth)

                loss = train_criterion(predict, targets)
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss_meter.update(loss.item())
            if i % 10 == 0:  # 每10个batch记录一次
                logger.info(
                    f'Epoch [{ep + 1}/{cfg["epochs"]}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        scheduler.step()

        # val
        with torch.no_grad():
            model.eval()  #告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].cuda()
                    label = sample['label'].cuda()
                    predict = model(image)[0]
                else:
                    image = sample['image'].cuda()
                    depth = sample['depth'].cuda()
                    label = sample['label'].cuda()
                    predict = model(image, depth)[0]


                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [b,c,h,w] to [c, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)


        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        writer.add_scalar("Val/mIoU_FRENet",test_miou,ep)
        test_f1 = running_metrics_test.get_scores()[0]["F1-Score: "]
        test_avg = (test_f1+ test_miou) / 2

        # 每轮训练结束后打印结果
        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'mPA={test_macc:.3f}, '
                    f'miou={test_miou:.3f}, '
                    f'avg={test_avg:.3f}')
        if test_avg > best_test and test_macc > 0.100 and test_miou > 0.100:
            best_test = test_avg
            save_ckpt(logdir, model)
            best_epo = ep + 1
            best_miou = test_miou
            best_avg = test_avg
            logger.info(
                '------------------------------------------------------------------------------------------------')
            logger.info(
                f'New result | [{best_epo}]  miou={best_miou:.3f}, avg={best_avg:.3f}')
            logger.info(
                '------------------------------------------------------------------------------------------------')
        else:
            logger.info(
                f'Best epo | [{best_epo}] , miou={best_miou:.3f}, avg={best_avg:.3f}')

        if ep >= 0.9 * cfg["epochs"]:
            name = f"{ep+1}" + "_"
            save_ckpt(logdir, model, name)

        if (ep + 1) % 50 == 0:
            name = f"{ep + 1}" + "_"
            save_ckpt(logdir, model, name)
    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/1SUS/configs/SUS_RGB.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpo int for continuing training")
    parser.add_argument("--gpu_ids", type=str, default='0', help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
