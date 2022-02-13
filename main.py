from configs.default import get_cfg_defaults
from torch.utils.data import DataLoader
import argparse

from utils.dataloader import DIV2KDataset
from pipelines import gan_pipeline, model_pipeline

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', type=str, default="EDSR.yaml", help='yaml configuration file')
    opts = parser.parse_args()

    return opts
    

if __name__ == "__main__":

    opts = get_args()
    config_file = f"./configs/{opts.config_file}"

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print(cfg)

    if cfg.DATASET.TRAIN_DATASET== "DIV2K":
        dataset = DIV2KDataset( cfg )
    else:
        raise NotImplementedError()


    dataloader = DataLoader( dataset, batch_size = cfg.DATASET.BATCH_SIZE, shuffle = True, pin_memory = True )

    if cfg.MODEL.GAN_MODEL:
        gan_pipeline( cfg, dataloader )
    else:
        model_pipeline( cfg, dataloader )
    
