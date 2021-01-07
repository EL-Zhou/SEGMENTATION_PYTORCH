# SEGMENTATION_PYTORCH

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.5
* CUDA 9.2 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/yiren-jian/SEGMENTATION_PYTORCH
$ cd SEGMENTATION_PYTORCH
```

1. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```

2. Optional. To uninstall this package, run:
```bash
$ pip uninstall SEGMENTATION_PYTORCH
```

## Dataset preparation
It assumes a preprocessed dataset with images and labels in numpy format.
- __numpy\_dataset__
   - __train__
     - __image__
       - [0.npy]
       - [1.npy]
       - [2.npy]
       - [3.npy]
       - [4.npy]
       - [5.npy]
     - __label__
       - [0.npy]
       - [1.npy]
       - [2.npy]
       - [3.npy]
       - [4.npy]
       - [5.npy]

The following gives an example of converting Cityscapes image dataset to Cityscapes numpy dataset.
```bash
$ python <root_dir>/segmentation_pytorch/datasets/preprocess_example.py
```
You should construct you custom dataset in the similar way.


## Running the code
### Training
To train Segmenattion model on Cityscapes dataset:
```bash
$ cd <root_dir>/segmentation_pytorch/scripts
$ python train.py --cfg ./configs/fcn.yml
```

### Testing
To test Segmenattion model on Cityscapes dataset:
```bash
$ cd <root_dir>/segmentation_pytorch/scripts
$ python test.py --cfg ./configs/fcn.yml
```

## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT).
