"""
Sasank Desaraju
9/21/2022
"""

#from asyncio.log import logger
from datetime import datetime
from importlib import import_module
from unicodedata import name
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_pose_hrnet import MyLightningModule, PoseHighResolutionNet
from lit_datamodule import MyLightningDataModule
import utility
#import click
import sys
import os
import time
import wandb

#torch.manual_seed(42) # haha HG2G



def main(config, wandb_run):
    data_module = MyLightningDataModule(
        config=config)
    pose_hrnet = PoseHighResolutionNet(num_key_points=1, num_image_channels=config.module['NUM_IMAGE_CHANNELS'])
    #model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run).load_from_checkpoint(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')
    model = MyLightningModule.load_from_checkpoint(CKPT_DIR + config.init['MODEL_NAME'] + '.ckpt', pose_hrnet=pose_hrnet, wandb_run=wandb_run)
    #model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run)
    #model = model.load_from_checkpoint(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')

    """
    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                        mode='min',
                                                        dirpath='checkpoints/',
                                                        filename=wandb_run.name)
    """

    trainer = pl.Trainer(accelerator='gpu',
        devices=-1,
        #logger=wandb_logger,
        default_root_dir=os.getcwd(),
        #callbacks=[save_best_val_checkpoint_callback],
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'])
    trainer.test(model, data_module)

if __name__ == '__main__':
    #"""
    # parsing the config
    CONFIG_DIR = os.getcwd() + '/config/'
    sys.path.append(CONFIG_DIR)
    config_module = import_module(sys.argv[1])
    #config_module = import_module('config/config')
    config = config_module.Configuration()
    #"""

    CKPT_DIR = os.getcwd() + '/checkpoints/'

    # setting up the logger
    os.environ["WANDB_RUN_GROUP"] = config.init['WANDB_RUN_GROUP']
    wandb_run = wandb.init(
        project=config.init['PROJECT_NAME'],
        name=config.init['RUN_NAME'],
        job_type='test'
    )

    main(config,wandb_run)

    wandb_run.finish()
