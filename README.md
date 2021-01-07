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
