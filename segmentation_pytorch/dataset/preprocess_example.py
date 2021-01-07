from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data

from segmentation_pytorch.utils.serialization import json_load

import numpy as np
import random
import os

def get_metadata(root, name, set):
    img_file = os.path.join(root, 'leftImg8bit', set, name)
    label_name = name.replace("leftImg8bit", "gtFine_labelIds")
    label_file = os.path.join(root, 'gtFine', set, label_name)
    return img_file, label_file

def get_image(image_size, file):
    return _load_img(file, image_size, Image.BICUBIC, rgb=True)

def get_labels(labels_size, file):
    return _load_img(file, labels_size, Image.NEAREST, rgb=False)

def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)

def map_labels(input_):
    info = json_load('/home/yiren/SEGMENTATION_PYTORCH/segmentation_pytorch/dataset/cityscapes_list/info.json')
    mapping = np.array(info['label2train'], dtype=np.int)
    map_vector = np.zeros((mapping.shape[0],), dtype=np.int64)
    for source_label, target_label in mapping:
        map_vector[source_label] = target_label
    return map_vector[input_.astype(np.int64, copy=False)]

def preprocess(mean, image):
    image = image[:, :, ::-1]  # change to BGR
    image -= mean
    return image.transpose((2, 0, 1))

def convert_img_to_numpy_and_save(set='train'):
    root = Path('/home/yiren/datasets/UDA/Cityscapes')
    set = set
    list_path = '/home/yiren/SEGMENTATION_PYTORCH/segmentation_pytorch/dataset/cityscapes_list/{}.txt'.format(set)
    image_size = (512, 256)
    labels_size = image_size
    mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    with open(list_path) as f:
        img_ids = [i_id.strip() for i_id in f]
    files = []
    for name in img_ids:
        img_file, label_file = get_metadata('/home/yiren/datasets/UDA/Cityscapes', name, set)
        files.append((img_file, label_file, name))

    saving_dir = '/home/yiren/cityscapes_numpy/{}'.format(set)
    for index in range(len(files)):
        img_file, label_file, name = files[index]
        label = get_labels(labels_size, label_file)
        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)

        label = map_labels(label).copy()
        image = get_image(image_size, img_file)
        image = preprocess(mean, image)

        image, label = image.copy(), label
        with open(os.path.join(saving_dir, 'image', '{}.npy'.format(index)), 'wb') as f:
            np.save(f, image)
        with open(os.path.join(saving_dir, 'label', '{}.npy'.format(index)), 'wb') as f:
            np.save(f, label)


if __name__ == '__main__':
    convert_img_to_numpy_and_save()
