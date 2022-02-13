from yacs.config import CfgNode as CN
import torch

_C = CN()


_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_C.SYSTEM.NUM_WORKERS = 4   


_C.DATASET = CN()
_C.DATASET.TRAIN_DATASET = "DIV2K"
_C.DATASET.TRAIN_DATASET_LENGTH = 800
_C.DATASET.TRAIN_DATASET_DIR = f"./datasets/{_C.DATASET.TRAIN_DATASET}"
_C.DATASET.TRAIN_PATCH_SIZE = 96
_C.DATASET.SR_SCALE = 2
_C.DATASET.BATCH_SIZE = 2
_C.DATASET.EVAL_DATASET = "Set5"
_C.DATASET.EVAL_DATASET_DIR = f"./datasets/{_C.DATASET.EVAL_DATASET}"


_C.MODEL = CN()
_C.MODEL.NAME = "RCAN"
_C.MODEL.PRETRAINED = True
_C.MODEL.GAN_MODEL = False


_C.TRAINER = CN()
# if not a gan model, this lr is used
_C.TRAINER.LR = 0.0001
# if it is a gan model, these learning rates are used
_C.TRAINER.GEN_LR = 0.0001
_C.TRAINER.DISC_LR = 0.0001
_C.TRAINER.EPOCHS = 400
# whether or not to write to tensorboard
_C.TRAINER.WRITE_TO_TB = False



def get_cfg_defaults():
    return _C.clone()