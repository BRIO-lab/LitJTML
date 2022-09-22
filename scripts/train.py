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
import utility
#import click
import sys
import os
import time
import wandb

#torch.manual_seed(42) # haha HG2G


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
    job_type='train'
    #id=str(time.strftime('%Y-%m-%d-%H-%M-%S'))     # this can be used for custom run ids but must be unique
    #dir='logs/'
    #save_dir='/logs/'
)

#wandb_logger = WandbLogger(log_model='all', save_dir='/logs/')
#wandb_logger = WandbLogger(log_model='all')
#wandb_logger.log


# define train/val sets
# perform sanity check for the sets (maybe this goes in a dataset file or a utility file?)

# log devices in use, their memory usage, etc.

data_module = MyLightningDataModule(
    #train_data='/blue/banks/sasank.desaraju/Sasank_JTML_seg/data/3_2_22_fem/train_3_2_22_fem.csv',
    config=config)
pose_hrnet = PoseHighResolutionNet(num_key_points=1, num_image_channels=config.module['NUM_IMAGE_CHANNELS'])
model = MyLightningModule(pose_hrnet=pose_hrnet, wandb_run=wandb_run) # I can put some data module stuff in this argument if I want

save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                    mode='min',
                                                    dirpath='checkpoints/',
                                                    filename=wandb_run.name)
trainer = pl.Trainer(accelerator='gpu',
    devices=-1,
    #logger=wandb_logger,
    default_root_dir=os.getcwd(),
    callbacks=[save_best_val_checkpoint_callback],
    fast_dev_run=config.init['FAST_DEV_RUN'],
    max_epochs=1,
    max_steps=2)
trainer.fit(model, data_module)
trainer.save_checkpoint(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')
wandb_run.save(CKPT_DIR + config.init['RUN_NAME'] + '.ckpt')