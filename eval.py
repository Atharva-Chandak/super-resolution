import cv2
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from tqdm import tqdm  


class Eval:
    
    def __init__(self, dataset, cfg ):

        self.eval_dataset = dataset
        self.device = cfg.SYSTEM.DEVICE
        self.data_loader = DataLoader( dataset, batch_size=1 , shuffle = False, pin_memory = True )

    def evaluate(self,model):
        print(">>>> Evaluating")
        model.eval()
        self.eval_psnrs = []
        
        for i,imgs in enumerate(tqdm(self.data_loader)):
            
            lr_eval_img = imgs['lr_eval_img'][:,:,:96,:96]
            hr_eval_img = imgs['hr_eval_img'][:,:,:96*2,:96*2]
            
            test_out = model(lr_eval_img.to(self.device).float())
            self.eval_psnrs.append(cv2.PSNR(test_out.permute(0, 2,  3, 1)[0].cpu().detach().numpy().astype(np.uint8),hr_eval_img.permute(0, 2,  3, 1)[0].cpu().detach().numpy().astype(np.uint8)))
            _, axis = plt.subplots(1,2)
            axis[0].imshow(np.uint8(hr_eval_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0]))
            axis[1].imshow(np.uint8(test_out.permute(0, 2, 3, 1).detach().cpu().numpy()[0]))
            plt.show()

        self.mean_psnr = sum(self.eval_psnrs)/len(self.eval_psnrs)
        
        return self.mean_psnr