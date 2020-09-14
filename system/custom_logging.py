from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from loguru import logger
import pytorch_lightning as pl
import torch
import sys
import os
from tqdm import tqdm


class LatestCheckpoint(Callback):
    """ save latest checkpoint """
    def __init__(self, ckpt_path, period=1, verbose=False):
        super().__init__()
        self.period = period
        self.ckpt_path = ckpt_path
        self.verbose = verbose

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.period == 0:
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path, exist_ok=True)
            path_to_save = os.path.join(self.ckpt_path, "latest.ckpt")
            trainer.save_checkpoint(path_to_save)

            if self.verbose:
                logger.debug(f"SAVE CHECKPOINT : {path_to_save}")


class BestCheckpoint(Callback):
    """ save latest checkpoint """
    def __init__(self, ckpt_path, best_value=-1, period=1, verbose=False):
        super().__init__()
        self.period = period
        self.ckpt_path = ckpt_path
        self.verbose = verbose
        self.best_value = best_value
        self.path_to_save = os.path.join(self.ckpt_path, "best.ckpt")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.period == 0:
            if len(trainer.logger.metrics
                   ) > 0 and trainer.logger.metrics[-1] > self.best_value:
                trainer.save_checkpoint(self.path_to_save)
                self.best_value = trainer.logger.metrics[-1]
                if self.verbose:
                    logger.debug(
                        f"SAVE CHECKPOINT : {self.path_to_save} for value {trainer.logger.metrics[-1]:.4f}"
                    )


class CustomLogger(LightningLoggerBase):
    def __init__(self, config=None):
        super().__init__()
        self.logger = logger
        self.config = config

        self.logger.remove()
        self.logger.configure(handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=''),
                level='DEBUG',
                colorize=True,
                format=
                "<green>{time: MM-DD at HH:mm}</green>  <level>{message}</level>"
            ),
        ])

        # add file handler for training mode
        if not config.test and not config.disable_logfile:
            self.logger.info(f"Log to file {config.log_path}")
            self.logger.add(sink=config.log_path,
                            mode='w',
                            format="{time: MM-DD at HH:mm} | {message}",
                            level="DEBUG")

        self.logger.info("\n" + str(config))

    @property
    def experiment(self):
        return self.logger

    def log_metrics(self, metrics, step=None):
        _str = ""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            _str += f"{k}: {v: .4f}  "
        if _str:
            self.logger.info(_str)

    def info_metrics(self, metrics, epoch=None, step=None):
        if isinstance(metrics, str):
            self.logger.info(metrics)
            return
        _str = ""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            _str += f"{k}: {v:.3f}  "
        self.logger.info(f"epoch {epoch: <4d}: iter {step:<6d}:: {_str}")

    @property
    def name(self):
        if self.config is None:
            return "log"
        else:
            return self.config.model_name

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params):
        return super().log_hyperparams(params)