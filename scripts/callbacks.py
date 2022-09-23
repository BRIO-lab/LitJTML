"""
Sasank Desaraju
9/23/22
This is to handle callbacks to keep our code clean and nice."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

class JTMLCallback(Callback):
    def __init__(self, config, wandb_run) -> None:
    #def __init__(self) -> None:
        super().__init__()

        self.config = config
        self.wandb_run = wandb_run
        print(self.config.init['MODEL_NAME'])

    """
    *********************** Init ***********************
    """

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        print(20 * '*' + "  Starting Initialization!  " + 20 * '*')

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        print(20 * '*' + "  Finished Initialization!  " + 20 * '*')
        return super().on_init_end(trainer)

    """
    *********************** Fit ***********************
    """

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Fitting!  " + 20 * '*')
        return super().on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Fitting!  " + 20 * '*')
        return super().on_fit_end(trainer, pl_module)
    
    """
    *********************** Fit/Train ***********************
    """

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Training!  " + 20 * '*')
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Training!  " + 20 * '*')
        return super().on_train_end(trainer, pl_module)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_epoch_end(trainer, pl_module)

    """
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_batch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_batch_end(trainer, pl_module)
    """

    """
    *********************** Fit/Validation ***********************
    """

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Validation!  " + 20 * '*')
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Validation!  " + 20 * '*')
        return super().on_validation_end(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #self.wandb_run.log
        return super().on_validation_epoch_end(trainer, pl_module)

    """
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    """

    """
    *********************** Test ***********************
    """

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Starting Testing!  " + 20 * '*')
        return super().on_test_start(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(20 * '*' + "  Finished Testing!  " + 20 * '*')
        return super().on_test_end(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_test_epoch_end(trainer, pl_module)

    """
    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    """










"""
    save_best_val_checkpoint_callback = ModelCheckpoint(monitor='validation/loss',
                                                        mode='min',
                                                        dirpath='checkpoints/',
                                                        filename=wandb_run.name)
"""