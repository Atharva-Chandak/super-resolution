import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    
    def __init__(self, cfg ):
        
        self.device = cfg.SYSTEM.DEVICE
        if cfg.TRAINER.WRITE_TO_TB:
            self.writer = SummaryWriter()
        else:
            self.writer = None
        if cfg.MODEL.NAME in ["EDSR","RCAN"]:
            self.train = self.simple_trainer
        elif cfg.MODEL.NAME in ["SRGAN"]:
            self.train = self.gan_trainer
        else:
            raise NotImplementedError("Trainer for this model is not defined")
    
    def simple_trainer(self, model, dataloader, optimizer, scheduler, cfg):
        self.model = model
        self.model.train()
        self.l1_criterion = nn.L1Loss()
        
        for e in range( cfg.TRAINER.EPOCHS ):
            
            for batch_id, batch in enumerate(tqdm(dataloader)):

                (lr_batch, hr_batch) = batch['lr_img'].to(self.device).float(), batch['hr_img'].to(self.device).float()
                
                optimizer.zero_grad()
                sr_batch = model( lr_batch )
                l1_loss = self.l1_criterion( sr_batch, hr_batch )
                self.loss = l1_loss

                if batch_id % 10 == 0:
                    tqdm.write(f"[Epoch]:{e}\t [Batch_id]:{batch_id} Loss: {self.loss.item()} ")

                self.writer.add_scalar('Loss/loss', self.loss.item(), len(dataloader)*e + batch_id )
                self.loss.backward()
                optimizer.step()

            scheduler.step()
    
    def gan_trainer( self, epochs, optimizer, dataloader, ):
        pass

    # def train(self,):
    #     self.model.train()

