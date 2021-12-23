"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import sys, os

from seaborn.matrix import heatmap
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil
import  traceback

import logging
import time
import timeit
from PIL import Image
import cv2

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
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
    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/best.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    # args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    gpus = list(config.GPUS)
    gpu_env_str = ''
    for i, gpu in enumerate(gpus):
        if i > 0: gpu_env_str += ','
        gpu_env_str += str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_env_str

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream

    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda()

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))
    model = FullModel(model, criterion)
    checkpoint = torch.load(model_state_file)
    model.model.load_state_dict(checkpoint['state_dict'])
    print("Epoch: {}".format(checkpoint['epoch']))
    model = nn.DataParallel(model).cuda()

    dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)
    (dataset_paths['SAVE_PRED'] / 'heatmap').mkdir(parents=True, exist_ok=True)


    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            path = dataset_list[it].get_tamp_name(i)
            name = os.path.split(path)[-1]
            return name, path

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred_binary = ((pred>0.5).byte()*255).cpu().numpy()
            pred = pred.cpu().numpy()

            # filename
            filename = os.path.splitext(get_next_filename(index)[0])[0] + ".png"
            filepath = dataset_paths['SAVE_PRED'] / filename
            heatmappath = dataset_paths['SAVE_PRED'] / 'heatmap' / filename

            # plot
            try:
                img = Image.fromarray(pred_binary)
                img.save(filepath)
                # width = pred.shape[1]  # in pixels
                # fig = plt.figure(frameon=False)
                # dpi = 40  # fig.dpi
                # fig.set_size_inches(width / dpi, ((width * pred.shape[0])/pred.shape[1]) / dpi)
                # sns.heatmap(pred, vmin=0, vmax=1, cbar=False, cmap='jet', )
                # plt.axis('off')
                # plt.savefig(heatmappath, bbox_inches='tight', transparent=True, pad_inches=0)
                # plt.close(fig)
                ori_img = np.float32(Image.open(get_next_filename(index)[1])) / 255
                heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(pred*255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                heatmap = np.float32(cv2.resize(heatmap, dsize=ori_img.shape[:2], interpolation=cv2.INTER_LINEAR)) / 255
                heatmap = Image.fromarray(np.uint8((heatmap + ori_img) / 2 * 255))
                heatmap.save(heatmappath)
            except:
                print(f"Error occurred while saving output. ({get_next_filename(index)[1]})")
                traceback.print_exc()

if __name__ == '__main__':
    main()
