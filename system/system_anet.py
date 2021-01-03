import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import argparse

import data_loader
from utils import init_weights
from .tester import Tester


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams.act_thresh = np.linspace(
            self.hparams.act_thresh_params[0],
            self.hparams.act_thresh_params[1],
            self.hparams.act_thresh_params[2])
        self.hparams.tIoU_thresh = np.arange(*self.hparams.tIoU_thresh_params)

        self.net = HAMNet(hparams)

        self.tester = Tester(self.hparams)

    def forward(self, x):
        return self.net(x, include_min=self.hparams.adl_include_min == 'true')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         conflict_handler="resolve")
        parser.add_argument("--model_name", type=str, default="dev0.5_anet")
        parser.add_argument("--rat", type=int, default=20, help="topk value")
        parser.add_argument("--rat2", type=int, default=5, help="topk value")

        parser.add_argument("--beta", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--num_segments", type=int, default=80)
        parser.add_argument("--percent_sup", type=float, default=0.1)
        parser.add_argument("--sampling", type=str, default="random")
        parser.add_argument('--class_thresh', type=float, default=0.2)
        # NOTE: work on anet dataset
        parser.add_argument("--dataset_name",
                            type=str,
                            default="ActivityNet1.2")
        parser.add_argument("--num_class", type=int, default=100)

        parser.add_argument("--scale", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-5)

        parser.add_argument("--lm_1", type=float, default=0.25)
        parser.add_argument("--lm_2", type=float, default=2)
        parser.add_argument('--gamma-oic', type=float, default=0.2)

        parser.add_argument("--drop_thres", type=float, default=0.2)
        parser.add_argument("--drop_prob", type=float, default=0.5)
        parser.add_argument('--adl_include_min', type=str, default='true')

        parser.add_argument("--batch_size", type=float, default=20)

        parser.add_argument("--min_percent", type=float, default=1.0)
        parser.add_argument("--tIoU_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.5, 1, 0.05])
        parser.add_argument("--act_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.9, 10])

        parser.add_argument("--rand", type=str, default='false')
        parser.add_argument("--max_epochs", type=int, default=20)
        return parser

    # --------------------------------- load data -------------------------------- #
    def train_dataloader(self):
        dataset = data_loader.Dataset(self.hparams,
                                      mode='train',
                                      sampling=self.hparams.sampling)
        if self.logger is not None:
            self.logger.experiment.info(
                f"Total training videos: {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = data_loader.Dataset(self.hparams,
                                      mode='test',
                                      sampling='all')
        if self.logger is not None:
            self.logger.experiment.info(
                f"Total testing videos: {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers)
        self.class_dict = dataset.class_dict
        return dataloader

    # ---------------------------------- config ---------------------------------- #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=0.001)

        return [optimizer]

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        total_loss, tqdm_dict = self._loss_fn(batch)
        self.log_dict(tqdm_dict, prog_bar=False)
        return total_loss

    # ----------------------------------- test ----------------------------------- #
    def test_step(self, batch, batch_idx):
        self.tester.eval_one_batch(batch, self, self.class_dict)

    def test_epoch_end(self, outputs):
        mAP = self.tester.final(logger=self.logger.experiment,
                                class_dict=self.class_dict)

        mAP = torch.tensor(mAP).to(self.device)
        self.log("mAP", mAP, prog_bar=True)

        self.tester = Tester(self.hparams)

    # ------------------------------ loss functions ------------------------------ #
    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _loss_fn(self, batch):
        """ Total loss funtion """
        features, labels, segm, vid_name, _ = batch

        element_logits, atn_supp, atn_drop, element_atn = self.net(features)

        element_logits_supp = self._multiply(element_logits, atn_supp)

        element_logits_drop = self._multiply(
            element_logits, (atn_drop > 0).type_as(element_logits),
            include_min=True)
        element_logits_drop_supp = self._multiply(element_logits,
                                                  atn_drop,
                                                  include_min=True)

        loss_1_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=self.hparams.rat,
                                       reduce=None)
        loss_1_drop, _ = self.topkloss(element_logits_drop,
                                       labels,
                                       is_back=True,
                                       rat=self.hparams.rat,
                                       reduce=None)

        loss_2_orig_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=self.hparams.rat,
                                            reduce=None)
        loss_2_drop_supp, _ = self.topkloss(element_logits_drop_supp,
                                            labels,
                                            is_back=False,
                                            rat=self.hparams.rat,
                                            reduce=None)

        wt = self.hparams.drop_prob

        loss_1 = (wt * loss_1_drop + (1 - wt) * loss_1_orig).mean()

        loss_2 = (wt * loss_2_drop_supp + (1 - wt) * loss_2_orig_supp).mean()

        elem_sort = element_atn.sort(dim=-2)[0]

        loss_norm = elem_sort[:, :int(self.hparams.num_segments *
                                      self.hparams.min_percent)].mean()

        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (self.hparams.lm_1 * loss_1 + self.hparams.lm_2 * loss_2 +
                      self.hparams.alpha * loss_norm +
                      self.hparams.beta * loss_guide)
        tqdm_dict = {
            "loss_train": total_loss,
            "loss_1": loss_1,
            "loss_2": loss_2,
            "loss_norm": loss_norm,
            "loss_guide": loss_guide,
        }

        return total_loss, tqdm_dict

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )

        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind


# ---------------------------------------------------------------------------- #
#                                     model                                    #
# ---------------------------------------------------------------------------- #
class HAMNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_class = args.num_class
        n_feature = args.feature_size

        if args.rat2 is not None:
            _kernel = ((args.num_segments // args.rat2) // 2 * 2 + 1)
        else:
            _kernel = None

        self.classifier = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.5), nn.Conv1d(n_feature, n_class + 1, 1),
            (nn.AvgPool1d(
                _kernel, 1, padding=_kernel // 2, count_include_pad=True)
             if _kernel is not None else nn.Identity()))

        self.attention = nn.Sequential(
            nn.Conv1d(n_feature, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 3,
                      padding=1), nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
            nn.Sigmoid(), (nn.AvgPool1d(
                _kernel, 1, padding=_kernel // 2, count_include_pad=True)
                           if _kernel is not None else nn.Identity()))
        self.adl = ADL(drop_thres=args.drop_thres, drop_prob=args.drop_prob)
        self.apply(init_weights)

    def forward(self, inputs, include_min=False):
        x = inputs.transpose(-1, -2)
        x_cls = self.classifier(x)
        x_atn = self.attention(x)

        atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=include_min)

        return x_cls.transpose(-1, -2), atn_supp.transpose(
            -1, -2), atn_drop.transpose(-1, -2), x_atn.transpose(-1, -2)


class ADL(nn.Module):
    def __init__(self, drop_thres=0.5, drop_prob=0.5):
        super().__init__()
        self.drop_thres = drop_thres
        self.drop_prob = drop_prob

    def forward(self, x, x_atn, include_min=False):
        if not self.training:
            return x_atn, x_atn

        # important mask
        mask_imp = x_atn

        # drop mask
        if include_min:
            atn_max = x_atn.max(dim=-1, keepdim=True)[0]
            atn_min = x_atn.min(dim=-1, keepdim=True)[0]
            _thres = (atn_max - atn_min) * self.drop_thres + atn_min
        else:
            _thres = x_atn.max(dim=-1, keepdim=True)[0] * self.drop_thres
        drop_mask = (x_atn < _thres).type_as(x) * x_atn

        return mask_imp, drop_mask
