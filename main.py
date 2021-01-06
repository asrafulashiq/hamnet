import torch
import pytorch_lightning as pl
from colorama import init
import os
import argparse
from options import parse_args
import utils
import system
from system.system_th import LightningSystem

init(autoreset=True)


def config_init():
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    # add Lightning parse
    parser = pl.Trainer.add_argparse_args(parser)

    # add common parse
    parser = parse_args(parser)

    # add model specific parser
    parser = LightningSystem.add_model_specific_args(parser)

    config = parser.parse_args()
    if config.seed >= 0:
        utils.set_seed(config.seed)
    config.model_name = config.model_name + config.suffix
    save_path = f"./ckpt/{config.model_name}/"
    config.save_path = save_path
    if config.test is False:
        os.makedirs(config.save_path, exist_ok=True)
    config.log_path = os.path.join(config.log_path, config.model_name + ".txt")
    config.gt_path = os.path.join('data', config.dataset_name, 'gt.json')
    return config


if __name__ == "__main__":
    config = config_init()
    model = LightningSystem(config)

    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt)['state_dict']
        model.load_state_dict(ckpt)

    trainer = pl.Trainer(
        logger=system.CustomLogger(config=config),
        default_root_dir=config.save_path,
        checkpoint_callback=False,
        gpus=config.gpus,
        max_epochs=config.max_epochs,
        progress_bar_refresh_rate=config.progress_refresh,
        callbacks=[system.LatestCheckpoint(config.save_path, verbose=False)])

    if config.test:
        trainer.test(model)
    else:
        trainer.fit(model)
        trainer.test(ckpt_path=None)
