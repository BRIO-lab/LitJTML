# let's try summ out


import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl

from pose_hrnet_modded_in_notebook import PoseHighResolutionNet

class MyLightningModule(pl.LightningModule):
    def __init__(self, pose_hrnet, wandb_run, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.pose_hrnet = pose_hrnet
        self.wandb_run = wandb_run
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        """This performs a forward pass on the dataset

        Args:
            x (this_type): This is a tensor containing the information yaya

        Returns:
            the forward pass of the dataset: using a certain type of input
        """
        return self.pose_hrnet(x)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        training_batch, training_batch_labels = train_batch['image'], train_batch['label']
        x = training_batch
        training_output = self.pose_hrnet(x)
        loss = self.loss_fn(training_output, training_batch_labels)
        #self.log('exp_train/loss', loss, on_step=True)
        #self.wandb_run.log('train/loss', loss, on_step=True)
        self.wandb_run.log({'train/loss': loss})
        return loss

    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        x = val_batch
        val_output = self.pose_hrnet(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        #self.log('validation/loss', loss)
        #self.wandb_run.log('validation/loss', loss, on_step=True)
        self.wandb_run.log({'validation/loss': loss})
        return loss

    def test_step(self, test_batch, batch_idx):
        test_batch, test_batch_labels = test_batch['image'], test_batch['label']
        x = test_batch
        test_output = self.pose_hrnet(x)
        loss = self.loss_fn(test_output, test_batch_labels)
        #self.log('test/loss', loss)
        #self.wandb_run.log('test/loss', loss, on_step=True)
        self.wandb_run.log({'test/loss': loss})
        return loss

"""
    def train_dataloader(self):
        return

    def val_dataloader(self):
        return
"""

    # def backward():
    # def optimizer_step():
