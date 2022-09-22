import torch
import torch.nn as nn
import albumentations as A
import numpy as np
import time

class Configuration:
    def __init__(self):
        self.init = {
            'PROJECT_NAME': 'LitJTML Development!',
            'RUN_NAME': 'Trying HPG Wandb Logging!' + time.strftime('%Y-%m-%d-%H-%M-%S'),
            'MODEL_NAME': 'MyModel'
        }
        self.etl = {
            'raw_data_file': "data/my_data.csv", \
            'processed_path': "data",\
            'VAL_SIZE':  0.2,       # looks sus
            'TEST_SIZE': 0.01,      # I'm not sure these two mean what we think
            #'random_state': np.random.randint(1,50)
            # HHG2TG lol; deterministic to aid reproducibility
            'random_state': 42,

            'CUSTOM_TEST_SET': False,
            'TEST_SET_NAME': '/my/test/set.csv'
        }

        self.dataset = {
            'IMAGE_HEIGHT': 1024,
            'IMAGE_WIDTH': 1024,
            'MODEL_TYPE': 'fem',        # how should we do this? not clear this is still best...
            'IMG_CHANNELS': 1
        }

        self.datamodule = {
            'IMAGE_DIRECTORY': '/blue/banks/sasank.desaraju/Sasank_JTML_seg/Images/TPLO_Ten_Dogs_grids_2_22_22',
            'BATCH_SIZE': 1,
            'SHUFFLE': True,
            'NUM_WORKERS': -1,
            'PIN_MEMORY': False,
            'SUBSET_PIXELS': True
        }

        self.module = {
            'LOSS_FN': nn.MSELoss()
        }

        # hyperparameters for training
        self.hparams = {
            'LOAD_FROM_CHECKPOINT': False,
            'learning_rate': 1e-3
        }

        self.transform = \
        A.Compose([
        A.RandomGamma(always_apply=False, p = 0.5,gamma_limit=(10,300)),
        A.ShiftScaleRotate(always_apply = False, p = 0.5,shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-180,180), interpolation=0, border_mode=0, value=(0, 0, 0)),
        A.Blur(always_apply=False, blur_limit=(3, 10), p=0.2),
        A.Flip(always_apply=False, p=0.5),
        A.ElasticTransform(always_apply=False, p=0.85, alpha=0.5, sigma=150, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
        A.InvertImg(always_apply=False, p=0.5),
        A.CoarseDropout(always_apply = False, p = 0.25, min_holes = 1, max_holes = 100, min_height = 25, max_height=25),
        A.MultiplicativeNoise(always_apply=False, p=0.25, multiplier=(0.1, 2), per_channel=True, elementwise=True)
    ], p=0.85)
