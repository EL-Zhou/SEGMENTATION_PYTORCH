import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from segmentation_pytorch.utils.func import adjust_learning_rate
from segmentation_pytorch.utils.func import loss_calc, bce_loss
from segmentation_pytorch.utils.viz_segmask import colorize_mask

from collections import OrderedDict


def loopy(dataloader):
    while True:
        for x in iter(dataloader): yield x


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_segmentation_model(model, dataloader, cfg):
    # Create the model and start the training.
    input_size_target = cfg.TRAIN.INPUT_SIZE
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    dataloader_iter = loopy(dataloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        model.adjust_learning_rate(optimizer, i_iter, cfg)

        # supervised training with cross entropy loss (on target cityscapes)
        batch = next(dataloader_iter)
        image, label, _, _ = batch
        _, pred = model(image.to(device))

        loss_seg = loss_calc(pred, label, None, device)
        loss_seg.backward()
        optimizer.step()

        current_losses = {'loss_seg': loss_seg}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, image, i_iter, pred, num_classes, 'T')
