import torch
from model_scripts import EDSR,RCAN

def get_model( cfg ):

    device = cfg.SYSTEM.DEVICE

    if cfg.MODEL.NAME == "EDSR":
        model = EDSR.EDSR( scale = cfg.DATASET.SR_SCALE)
        if cfg.MODEL.PRETRAINED:
            file_name = f'./pretrained/EDSR_trained_x{cfg.DATASET.SR_SCALE}.pt'
            model.load_state_dict(torch.load(file_name, map_location = device))
        return model.to(device)

    elif cfg.MODEL.NAME == "RCAN":
        model = RCAN.RCAN( scale = cfg.DATASET.SR_SCALE)
        if cfg.MODEL.PRETRAINED:
            file_name = f'pretrained/RCAN_trained_x{cfg.DATASET.SR_SCALE}.pt'
            model.load_state_dict(torch.load(file_name, map_location = device))
        return model.to(device)

    else:
        raise NotImplementedError("This model is not yet supported")    