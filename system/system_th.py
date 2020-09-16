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

        self.net = Model_BaS(hparams)

        self.best_metric = [-1, -1]
        self.tester = Tester(self.hparams)

    def forward(self, x):
        return self.net(x, include_min=self.hparams.adl_include_min == 'true')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         conflict_handler="resolve")
        parser.add_argument("--model-name", type=str, default="dev0.5_thumos")
        parser.add_argument("--rat", type=int, default=10, help="topk value")
        parser.add_argument("--rat2",
                            type=int,
                            default=None,
                            help="topk value")

        parser.add_argument("--beta", type=float, default=0.8)
        parser.add_argument("--alpha", type=float, default=0.8)
        parser.add_argument("--num-segments", type=int, default=500)
        parser.add_argument("--sampling", type=str, default="random")
        parser.add_argument('--class-thresh', type=float, default=0.2)
        # NOTE: work on anet dataset
        parser.add_argument("--dataset-name",
                            type=str,
                            default="Thumos14reduced")

        parser.add_argument("--scale", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-5)

        parser.add_argument("--lm_1", type=float, default=1)
        parser.add_argument("--lm_2", type=float, default=1)
        parser.add_argument("--gradient_clip_val", type=float, default=1)

        parser.add_argument("--drop-thres", type=float, default=0.2)
        parser.add_argument("--drop-prob", type=float, default=0.2)
        parser.add_argument('--gamma-oic', type=float, default=0.2)
        parser.add_argument('--adl-include-min', type=str, default='true')

        parser.add_argument("--batch-size", type=float, default=20)
        parser.add_argument("--percent-sup", type=float, default=0.1)

        parser.add_argument("--min-percent", type=float, default=1.0)
        parser.add_argument("--tIoU-thresh-params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.75, 0.1])
        parser.add_argument("--act-thresh-params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.9, 10])
        parser.add_argument("--loss-type", type=str,
                            default="softmax")  # ce/bce/softmax
        parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
        parser.add_argument("--save-pred", type=str, default=None)
        parser.add_argument("--rand", type=str, default='false')
        parser.add_argument("--max-epoch", type=int, default=100)
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
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
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
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        self.class_dict = dataset.class_dict
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    # ---------------------------------- config ---------------------------------- #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=0.001)
        return [optimizer]

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        total_loss, tqdm_dict = self._loss_fn(batch)

        result = pl.TrainResult(total_loss, checkpoint_on=False)
        result.log_dict(tqdm_dict, prog_bar=False, logger=True)
        return result

    # -------------------------------- validation -------------------------------- #
    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, mode='val')

    def validation_epoch_end(self, outputs):
        return self._validation_epoch_end(outputs, mode='val')

    def _validation_step(self, batch, batch_idx, mode='val'):
        self.tester.eval_one_batch(batch, self, self.class_dict)
        return {}

    def _validation_epoch_end(self, outputs, mode='val'):
        mAP = self.tester.final(logger=self.logger.experiment,
                                class_dict=self.class_dict)

        if mAP > self.best_metric[0]:
            self.best_metric = [mAP, self.current_epoch]

        self.logger.experiment.info(
            f"Best mAP: {self.best_metric[0]: .4f} at epoch {self.best_metric[1]}"
        )
        mAP = torch.tensor(mAP).to(self.device)
        result = pl.EvalResult(checkpoint_on=mAP)
        result.log("mAP", mAP, prog_bar=True)

        self.tester = Tester(self.hparams)
        return result

    # ----------------------------------- test ----------------------------------- #
    def test_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, mode='test')

    def test_epoch_end(self, outputs):
        return self._validation_epoch_end(outputs, mode='test')

    # ------------------------------ loss functions ------------------------------ #

    def _fn_mul(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _loss_fn(self, batch):
        """ Total loss funtion """
        features, labels, segm, vid_name, _ = batch

        element_logits, atn_supp, atn_drop, element_atn = self.net(features)

        element_logits_supp = self._fn_mul(element_logits, atn_supp)

        element_logits_drop = self._fn_mul(
            element_logits, (atn_drop > 0).type_as(element_logits),
            include_min=True)
        element_logits_drop_supp = self._fn_mul(element_logits,
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

        # loss_norm = torch.mean(torch.norm(element_atn, p=1, dim=-2))
        elem_sort = element_atn.sort(dim=-2)[0]

        loss_norm = elem_sort[:, :int(self.hparams.num_segments *
                                      self.hparams.min_percent)].mean()

        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # loss_ex = (-atn_supp * torch.log(atn_supp + 1e-5)).mean()
        # loss_ex = 0

        # total loss
        total_loss = (self.hparams.lm_1 * loss_1 + self.hparams.lm_2 * loss_2 +
                      self.hparams.alpha * loss_norm +
                      self.hparams.beta * loss_guide)
        #   self.hparams.gamma * loss_top +
        # self.hparams.lm_cont * loss_cont + self.hparams.lm_sup * loss_sup)
        tqdm_dict = {
            "loss_train": total_loss,
            "loss_1": loss_1,
            "loss_2": loss_2,
            "loss_1o": loss_1_orig.mean(),
            "loss_1d": loss_1_drop.mean(),
            "loss_2o": loss_2_orig_supp.mean(),
            "loss_2d": loss_2_drop_supp.mean(),
            # "loss_ex": loss_ex,
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

        if self.hparams.loss_type == "ce":
            instance_logits = instance_logits.softmax(dim=-1)
            wt = labels_with_back.sum(dim=-1,
                                      keepdim=True) / self.hparams.num_class
            eps = 1e-8
            milloss = (-(
                (1 - wt) * labels_with_back * torch.log(instance_logits + eps)
                + wt * (1 - labels_with_back) *
                torch.log(1 - instance_logits + eps)).sum(dim=-1)).mean()
        elif self.hparams.loss_type == "bce":
            labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
            instance_logits = instance_logits.softmax(dim=-1)
            milloss = F.binary_cross_entropy(instance_logits, labels_with_back)
        else:
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
class Model_BaS(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_class = args.num_class
        n_feature = args.feature_size

        if args.rat2 is not None:
            _kernel = ((args.num_segments // args.rat2) // 2 * 2 + 1)
        else:
            _kernel = None

        self.classifier = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_feature, n_feature, 3, padding=1),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(n_feature),
            nn.Dropout(0.7),
            # nn.Conv1d(n_feature, n_feature, 13, padding=6, groups=n_feature),
            # nn.LeakyReLU(0.2),
            nn.Conv1d(n_feature, n_class + 1, 1),
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

        # x_cls.clamp_(-5, 5)
        atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=include_min)

        return x_cls.transpose(-1, -2), atn_supp.transpose(
            -1, -2), atn_drop.transpose(-1, -2), x_atn.transpose(-1, -2)


class ADL(nn.Module):
    def __init__(self, drop_thres=0.5, drop_prob=0.5):
        super().__init__()
        self.drop_thres = drop_thres
        self.drop_prob = drop_prob

    def forward(self, x, x_atn, include_min=False):
        B, C, L = x.shape
        _min = x.min(dim=-2, keepdim=True)[0]

        if not self.training:
            return x_atn, x_atn
            # _rand = torch.zeros((B, 1)).type_as(x)
            # return (x - _min) * x_atn + _min, _rand

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
