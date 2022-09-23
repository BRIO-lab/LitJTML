"""
Sasank Desaraju
9/13/22
"""

import torch
import pytorch_lightning as pl
import numpy as np
import os
from skimage import io
import cv2

from lit_JTMLDataset import LitJTMLDataset


class MyLightningDataModule(pl.LightningDataModule):
    def __init__(self,
    config):
        super().__init__()

        self.train_data = config.temp['train_data']
        self.val_data = config.temp['val_data']
        self.test_data = config.temp['test_data']
        self.img_dir = config.datamodule['IMAGE_DIRECTORY']

        self.num_workers = config.datamodule['NUM_WORKERS']

        self.batch_size = config.datamodule['BATCH_SIZE']
        # other constants

        self.train_set = np.genfromtxt(self.train_data, delimiter=',', dtype=str)
        self.val_set = np.genfromtxt(self.val_data, delimiter=',', dtype=str)
        self.test_set = np.genfromtxt(self.test_data, delimiter=',', dtype=str)

        #check train dataset length and integrity
        #check val dataset length and integrity

    """
    def prepare_data(self):

        return
    """

    def setup(self, stage):
        # actually do all the stuff here I think

        """
        dataset = self.train_set

        if stage=='train' or stage is None:
            dataset = self.train_set
            #check dataset length and integrity

        if stage=='val' or stage is None:
            dataset = self.val_set

        created_dataset = LitJTMLDataset(dataset)
        """

        self.training_set = LitJTMLDataset(dataset=self.train_set, img_dir=self.img_dir)
        self.validation_set = LitJTMLDataset(dataset=self.val_set, img_dir=self.img_dir)
        self.test_set = LitJTMLDataset(dataset=self.test_set, img_dir=self.img_dir)

        return

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_set,
        batch_size=self.batch_size,
        num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_set,
        batch_size=self.batch_size,
        num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set,
        batch_size=self.batch_size,
        num_workers=self.num_workers)
