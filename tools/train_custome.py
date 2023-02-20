# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""

import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import gc
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader,random_split
from tensorboardX import SummaryWriter

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from Splicing.data.Defacto import DefactoDataset




def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    # args = parse_args()
    # Instead of using argparse, force these args:
    ## CHOOSE ##
    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', local_rank=0, opts=None)
    # args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', local_rank=0, opts=None)

    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        ## CHOOSE ##
        dataset = DefactoDataset(
                r"F:\datasets\Defacto_splicing\splicing_2_img\img_jpg",
                r"F:\datasets\Defacto_splicing\splicing_2_annotations\probe_mask",
                10764,
                512,
                'train',None,
                blocks=('RGB', 'DCTvol', 'qtable'))
        
        # DefactoDataset(crop_size=crop_size, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # full model
        # train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # only DCT stream
        logger.info(dataset.get_info())
    else:
        raise ValueError("Not supported dataset type.")
    
    # 수정시작
    print(dataset._crop_size)
    
    train_size =  10000 #int(dataset_size * 0.8)
    validation_size = 764 #int(dataset_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
   

    trainloader = DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True)
    validloader = DataLoader(validation_dataset, batch_size=2, num_workers=0, shuffle=False)
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())

    # validation
    ## CHOOSE ##
    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL).cuda()

    model = FullModel(model, criterion)

    # optimizer
    logger.info((f"# params with requires_grad = {len([c for c in model.parameters() if c.requires_grad])}\n",
                f"# params freezed = {len([c for c in model.parameters() if not c.requires_grad])}\n",
                '# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000))


    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int32(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_p_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_p_mIoU = checkpoint['best_p_mIoU']
            last_epoch = checkpoint['epoch']
            model.model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        else:
            logger.info("No previous checkpoint.")
    # print('test here')
    # return 0 
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # train
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict, final_output_dir)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # Valid
        if 1:
            print("Start Validating..")
            writer_dict['valid_global_steps'] = epoch
            valid_loss, mean_IoU, avg_mIoU, avg_p_mIoU, IoU_array, pixel_acc, mean_acc, confusion_matrix = \
                validate(config, validloader, model, writer_dict, "valid")

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3.0)

            if avg_p_mIoU > best_p_mIoU:
                best_p_mIoU = avg_p_mIoU
                torch.save({
                    'epoch': epoch + 1,
                    'best_p_mIoU': best_p_mIoU,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
                logger.info("best.pth.tar updated.")
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'best_p_mIoU': best_p_mIoU,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, f'chk_{epoch+1}.pth.tar'))
                logger.info("best.pth.tar updated.")
            msg = '(Valid) Loss: {:.3f}, MeanIU: {: 4.4f}, Best_p_mIoU: {: 4.4f}, avg_mIoU: {: 4.4f}, avg_p_mIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_p_mIoU, avg_mIoU, avg_p_mIoU, pixel_acc, mean_acc)
            logging.info(msg)
            logging.info(IoU_array)
            logging.info("confusion_matrix:")
            logging.info(confusion_matrix)

        else:
            logging.info("Skip validation.")

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'checkpoint.pth.tar')))
        torch.save({
            'epoch': epoch + 1,
            'best_p_mIoU': best_p_mIoU,
            'state_dict': model.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))


if __name__ == '__main__':
    main()
