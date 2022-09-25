"""
Sasank Desaraju
9/9/2022
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
from callbacks import JTMLCallback
#import click
import sys
import os
import time
import wandb

#torch.manual_seed(42) # haha HG2G





#def main(config, wandb_logger):
def main(config, wandb_run):
    data_module = MyLightningDataModule(config=config)
    pose_hrnet = PoseHighResolutionNet(num_key_points=1, num_image_channels=config.module['NUM_IMAGE_CHANNELS'])
    #model = MyLightningModule(pose_hrnet=pose_hrnet) # I can put some data module stuff in this argument if I want
    model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run) # I can put some data module stuff in this argument if I want

    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                        mode='min',
                                                        dirpath='checkpoints/',
                                                        filename=wandb_run.name)
                                                        #filename=wandb.run.name)
    trainer = pl.Trainer(accelerator='gpu',
        devices=-1,
        auto_select_gpus=True,
        #logger=wandb_logger,
        default_root_dir=os.getcwd(),
        callbacks=[JTMLCallback(config, wandb_run)],
        #callbacks=[JTMLCallback(config, wandb_run), save_best_val_checkpoint_callback],
        fast_dev_run=config.init['FAST_DEV_RUN'],
        max_epochs=config.init['MAX_EPOCHS'],
        max_steps=config.init['MAX_STEPS'],
        strategy=config.init['STRATEGY'])
    trainer.fit(model, data_module)
    trainer.save_checkpoint(CKPT_DIR + config.init['MODEL_NAME'] + '.ckpt')
    #wandb_run.save(CKPT_DIR + config.init['MODEL_NAME'] + '.ckpt')
    #wandb.save(CKPT_DIR + config.init['MODEL_NAME'] + time.strftime('%Y-%m-%d-%H-%M-%S') + '.ckpt')
    wandb.save(CKPT_DIR + config.init['MODEL_NAME'] + '.ckpt')

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
    os.environ['WANDB_RUN_GROUP'] = config.init['WANDB_RUN_GROUP']
    #wandb.init(
    wandb_run = wandb.init(
        project=config.init['PROJECT_NAME'],
        name=config.init['RUN_NAME'],
        group=config.init['WANDB_RUN_GROUP'],
        job_type='fit'
        #id=str(time.strftime('%Y-%m-%d-%H-%M-%S'))     # this can be used for custom run ids but must be unique
        #dir='logs/'
        #save_dir='/logs/'
    )
    #wandb_logger = WandbLogger()

    #main(config, wandb_logger)
    main(config, wandb_run)

    wandb.finish()
