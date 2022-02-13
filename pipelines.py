
from torch import optim
from model import get_model
from trainer import Trainer
from eval import Eval
from utils.dataloader import Set5Dataset


def gan_pipeline( cfg, dataloader ):

    gen_model, disc_model = get_model( cfg )

    gen_optimizer = optim.Adam( gen_model.parameters(), lr = cfg.TRAINER.GEN_LR )
    disc_optimizer = optim.Adam( disc_model.parameters(), lr = cfg.TRAINER.DISC_LR )
    gen_scheduler = optim.lr_scheduler.StepLR( gen_optimizer,  step_size = 5, gamma = 1 )
    disc_scheduler = optim.lr_scheduler.StepLR( disc_optimizer,  step_size = 5, gamma = 1 )

    if not cfg.MODEL.PRETRAINED:
        model_trainer = Trainer( cfg )
        model_trainer.train( gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler )

def model_pipeline( cfg, dataloader ):
    model = get_model( cfg )

    optimizer = optim.Adam( model.parameters(), lr = cfg.TRAINER.LR )
    scheduler = optim.lr_scheduler.StepLR( optimizer,  step_size = 5, gamma = 1 )

    # train if not using a pretrained model 
    if not cfg.MODEL.PRETRAINED:
        model_trainer = Trainer( cfg )
        model_trainer.train( model, dataloader, optimizer, scheduler, cfg )

    # Eval
    eval_dataset = Set5Dataset( cfg )
    evaluator = Eval( eval_dataset, cfg )
    mean_psnr = evaluator.evaluate( model )
    print(f"Mean PSNR of {eval_dataset} dataset:{mean_psnr:.4f}")
