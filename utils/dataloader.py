from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class DIV2KDataset(Dataset):

    def __init__( self, cfg ):

        self.dataset_dir = cfg.DATASET.TRAIN_DATASET_DIR
        self.scale = cfg.DATASET.SR_SCALE
        self.lr_data_dir = f'{self.dataset_dir}/DIV2K_train_LR_bicubic/X{self.scale}'
        self.hr_data_dir = f'{self.dataset_dir}/DIV2K_train_HR'
        self.patch_size = cfg.DATASET.TRAIN_PATCH_SIZE
        self.dataset_length = cfg.DATASET.TRAIN_DATASET_LENGTH

    def __len__(self):
        return self.dataset_length
    
    
    def patch(self, lr_img, hr_img, scale = None,  p_size = None ):
        if scale is None:
            scale = self.scale
        if p_size is None:
            p_size = self.patch_size
        y, x = random.randint(0, lr_img.shape[1] - p_size), random.randint(0, lr_img.shape[2] - p_size )
        lr_patch = lr_img[:, y : y + p_size, x : x + p_size]
        hr_patch = hr_img[:, y*scale : y*scale + p_size*scale, x*scale : x*scale + p_size*scale]

        return lr_patch, hr_patch 

    def __getitem__(self,idx):
        
        img_name =str(idx+1).zfill(4)
        lr_img_path = f'{self.lr_data_dir}/{img_name}x{self.scale}.png'
        hr_img_path = f'{self.hr_data_dir}/{img_name}.png'

        with open(lr_img_path, 'rb') as f:
            lr_img = transforms.PILToTensor()(Image.open(f).convert('RGB'))

        with open(hr_img_path, 'rb') as f:
            hr_img = transforms.PILToTensor()(Image.open(f).convert('RGB'))

        lr_patch,hr_patch = self.patch(lr_img,hr_img)

        return {
            'lr_img':lr_patch,
            'hr_img':hr_patch,
        }

class Set5Dataset(Dataset):

    def __init__( self, cfg ):

        self.dataset_dir = cfg.DATASET.EVAL_DATASET_DIR
        self.scale = cfg.DATASET.SR_SCALE
        self.lr_data_dir = f'{self.dataset_dir}/LR_bicubic/X{self.scale}'
        self.hr_data_dir = f'{self.dataset_dir}/HR'

        self.img_names = ['baby','bird','butterfly','head','woman']

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self,idx):
        
        img_name = self.img_names[idx]
        lr_img_path = f'{self.lr_data_dir}/{img_name}x{self.scale}.png'
        hr_img_path = f'{self.hr_data_dir}/{img_name}.png'

        with open(lr_img_path, 'rb') as f:
            lr_eval_img = transforms.PILToTensor()(Image.open(f).convert('RGB'))

        with open(hr_img_path, 'rb') as f:
            hr_eval_img = transforms.PILToTensor()(Image.open(f).convert('RGB'))

        return {
            'lr_eval_img':lr_eval_img,
            'hr_eval_img':hr_eval_img,
        }

