# let's try summ out


import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
import wandb

from pose_hrnet_modded_in_notebook import PoseHighResolutionNet

class SegmentationNetModule(pl.LightningModule):
    def __init__(self, pose_hrnet, wandb_run, learning_rate=1e-3):
    #def __init__(self, pose_hrnet, learning_rate=1e-3):
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
        #self.log(name="train/loss", value=loss)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        x = val_batch
        val_output = self.pose_hrnet(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        #self.log('validation/loss', loss)
        #self.wandb_run.log('validation/loss', loss, on_step=True)
        self.wandb_run.log({'validation/loss': loss})
        #self.log('validation/loss', loss)
        image = wandb.Image(val_output, caption='Validation output')
        self.wandb_run.log({'val_output': image})
        return loss

    def test_step(self, test_batch, batch_idx):
        test_batch, test_batch_labels = test_batch['image'], test_batch['label']
        x = test_batch
        test_output = self.pose_hrnet(x)
        loss = self.loss_fn(test_output, test_batch_labels)
        #self.log('test/loss', loss)
        #self.wandb_run.log('test/loss', loss, on_step=True)
        #self.wandb_run.log({'test/loss': loss})
        #self.on_test_batch_end(self, outputs=test_output, batch=test_batch, batch_idx=batch_idx)
        #self.on_test_batch_end(outputs=test_output, batch=test_batch, batch_idx=batch_idx, dataloader_idx=0)
        return loss
    

    #def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx) -> None:
    """
    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx, **kwargs) -> None:
        print(outputs.size())
        for image in outputs:
            image = self.wandb.Image(image, caption='Test output from batch ' + str(batch_idx))
            self.wandb_run.log({'test_output': image})
        #return super().on_test_batch_end(trainer, pl_module, batch, batch_idx)
        #return super().on_test_batch_end(batch, batch_idx)
    """

"""
    def train_dataloader(self):
        return

    def val_dataloader(self):
        return
"""

    # def backward():
    # def optimizer_step():