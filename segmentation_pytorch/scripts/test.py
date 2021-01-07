import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data

from segmentation_pytorch.model.deeplabv2 import get_deeplab_v2
from segmentation_pytorch.model.fcn8s_v2 import get_fcn8s_vgg
from segmentation_pytorch.dataset.cityscapes import CityscapesDataSet as MyDataSet
from segmentation_pytorch.segmentation.config import cfg, cfg_from_file
from segmentation_pytorch.segmentation.eval_segmentation import evaluate_segmentation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.TARGET}_{cfg.TRAIN.MODEL}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        elif cfg.TEST.MODEL[i] == 'FCN8s':
            model = get_fcn8s_vgg(num_classes=cfg.NUM_CLASSES)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)


    # dataloaders
    test_dataset = MyDataSet(root=cfg.DATA_DIRECTORY,
                             list_path=cfg.DATA_LIST,
                             set=cfg.TEST.SET,
                             info_path=cfg.TEST.INFO,
                             crop_size=cfg.TEST.INPUT_SIZE,
                             mean=cfg.TEST.IMG_MEAN,
                             labels_size=cfg.TEST.OUTPUT_SIZE)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    # eval
    evaluate_segmentation(models, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
