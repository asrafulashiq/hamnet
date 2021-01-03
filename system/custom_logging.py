from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from loguru import logger
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm
from pytorch_lightning.utilities.distributed import rank_zero_only
from colorama import init, Fore

init(autoreset=True)


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

    @property
    def experiment(self):
        return self.logger

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            try:
                return value.item()
            except ValueError:
                return value.mean().item()
        return value

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if len(metrics) == 0:
            return

        metrics_str = "  ".join([
            f"{k}: {self._handle_value(v):<4.4f}" for k, v in metrics.items()
            if k != 'epoch'
        ])

        if metrics_str.strip() == '':
            return

        if step is not None:
            metrics_str = f"step: {step:<6d} :: " + metrics_str
        if 'epoch' in metrics:
            metrics_str = f"epoch: {int(metrics['epoch']):<4d}  " + metrics_str
        self.experiment.info(metrics_str)

    @rank_zero_only
    def log_hyperparams(self, params):
        _str = ""
        for k in sorted(params):
            v = params[k]
            _str += Fore.LIGHTCYAN_EX + str(k) + "="
            _str += Fore.WHITE + str(v) + ", "
        self.experiment.info("\nhyper-parameters:\n" + _str + "\b\b" + "  ")
        return

    @property
    def name(self):
        if self.config is None:
            return "log"
        else:
            return self.config.model_name

    @property
    def version(self):
        return 0
