import argparse
import os
import os.path as osp
import pprint
import random
import warnings

import numpy as np
import yaml
import torch
from torch.utils import data

from segmentation_pytorch.model.deeplabv2 import get_deeplab_v2
from segmentation_pytorch.model.fcn8s_v2 import get_fcn8s_vgg
# from segmentation_pytorch.dataset.cityscapes import CityscapesDataSet as MyDataSet
from segmentation_pytorch.dataset.mydataset import MyDataSet
from segmentation_pytorch.segmentation.config import cfg, cfg_from_file
from segmentation_pytorch.segmentation.train_segmentation import train_segmentation_model

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for Semantic Segmentaion")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.TARGET}_{cfg.TRAIN.MODEL}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    args.tensorboard = True # always turn tensorboard on
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)


    # LOAD SEGMENTATION NET
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    elif cfg.TRAIN.MODEL == 'FCN8s':
        model = get_fcn8s_vgg(num_classes=cfg.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')


    train_dataset = MyDataSet(root=cfg.DATA_DIRECTORY,
                              list_path=cfg.DATA_LIST,
                              set=cfg.TRAIN.SET,
                              info_path=cfg.TRAIN.INFO,
                              max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE,
                              crop_size=cfg.TRAIN.INPUT_SIZE,
                              mean=cfg.TRAIN.IMG_MEAN,
                              random_horizontal_flip=True)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=cfg.NUM_WORKERS,
                                   shuffle=True,
                                   pin_memory=True,
                                   worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # SEGMENTATION TRAINING
    train_segmentation_model(model, train_loader, cfg)


if __name__ == '__main__':
    main()
