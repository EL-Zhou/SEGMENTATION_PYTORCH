import numpy as np
import random
import os
import fnmatch
import torch

from torch.utils.data.dataset import Dataset

from segmentation_pytorch.utils import project_root
from segmentation_pytorch.utils.serialization import json_load
from segmentation_pytorch.dataset.base_dataset import BaseDataset


class RandomHorizontalFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = np.flip(image,axis=-1)
            label = np.flip(label,axis=-1)
        return image.copy(), label.copy()


class MyDataSet(Dataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=None, labels_size=None,
                 random_horizontal_flip=False):

        self.root = os.path.expanduser(root)
        self.augmentation = random_horizontal_flip
        train = True if set == 'train' else False

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = np.load(self.data_path + '/image/{:d}.npy'.format(index))
        label = np.load(self.data_path + '/label/{:d}.npy'.format(index))

        # apply data augmentation if required
        if self.augmentation is True:
            image, label = RandomHorizontalFlip()(image, label)
            image, label = image.copy(), label.copy()

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label, image.size(), index

    def __len__(self):
        return self.data_len
