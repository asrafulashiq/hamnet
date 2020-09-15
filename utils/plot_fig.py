from fileinput import filename
import json
import matplotlib
from matplotlib.pyplot import xlim
import numpy as np
import pandas as pd
# from config import class_dict
from scipy.interpolate import interp1d
import os
import torch
import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque
import seaborn as sns
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as sio
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages
import random
from colorama import Fore
from loguru import logger
import shutil


def get_pred_loc(x, threshold=0.1):
    pred_loc = []
    vid_pred = np.concatenate(
        [np.zeros(1), (x > threshold).astype('float32'),
         np.zeros(1)], axis=0)
    vid_pred_diff = [
        vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
    ]
    s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
    e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
    for j in range(len(s)):
        if e[j] - s[j] >= 2:
            pred_loc.append((s[j], e[j]))
    return pred_loc


def plot_vspan(Y, ax, ymin=0, ymax=1, color=(1, 0, 0)):
    for i in range(len(Y) - 1):
        xs = i
        xe = i + 1
        clr = list(color) + [Y[i]]
        ax.axvspan(xmin=xs,
                   xmax=xe + 2,
                   ymin=ymin,
                   ymax=ymax,
                   color=tuple(clr))


def interpolate_logit(arr, tlen):
    x = np.linspace(0, tlen, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, tlen)
    up_scale = f(scale_x)
    return up_scale


class PlotCam():
    matplotlib.rcParams['axes.linewidth'] = 0.5

    def __init__(self, filename, class_dict, plot_per_page=5, separate=False):
        plt.tick_params(axis="both", which="minor", labelsize="small")
        plt.rcParams.update({"font.size": 5})

        self.separate = separate

        if not self.separate:
            sns.set_style("white")
            filename = filename + ".pdf"
            self.pdf = PdfPages(filename)
            self.plot_per_page = plot_per_page
            self.counter = 0
            _f, _a = plt.subplots(plot_per_page, 1)
            self.cur_fig = _f
            self.cur_axes = _a.tolist()
            logger.info(f"PLOT::writing to pdf file {filename}...")
        else:
            if os.path.exists(filename):
                shutil.rmtree(filename)
            os.makedirs(filename, exist_ok=True)
            self.folder_name = filename
        self.class_dict = class_dict

    @staticmethod
    def plot_line(pred, gt, title, ax, color='red'):
        x = np.arange(len(gt))
        ax.fill_between(x, gt, color="green", linewidth=1, alpha=0.2)
        ax.plot(x, pred, linewidth=1, linestyle="-", color=color)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, len(pred))
        ax.grid(True,
                axis="y",
                color="grey",
                lw=0.1,
                alpha=0.2,
                linestyle="--")

    @staticmethod
    def plot_line_vspan(pred,
                        gt,
                        title,
                        ax: plt.Axes,
                        color=(0, 0, 0),
                        thres=0.5):
        x = np.arange(len(gt))
        _len = len(gt)

        # _thres = max(pred.max() * thres, 0.3)
        _thres = max(pred.max() * 0.7, 0.3)
        predt = (pred > _thres).astype(float)

        preds = pred * (0.45 - 0.05) + 0.05

        ax.fill_between(x,
                        preds,
                        y2=0.05,
                        linewidth=0.50,
                        linestyle="-",
                        color=color,
                        alpha=0.4)
        ax.hlines(0.50, 0, x[-1], color='k', lw=0.5)
        ax.hlines(1.0, 0, x[-1], color='k', lw=0.5)
        for i in range(1, _len - 2):
            ax.axvspan(xmin=x[i],
                       xmax=x[i + 1],
                       ymin=1. / 3 * 1.05,
                       ymax=2. / 3 * 0.95,
                       color=color,
                       alpha=predt[i])
            if gt[i] == 1:
                ax.axvspan(xmin=x[i],
                           xmax=x[i + 1],
                           ymin=2. / 3 * 1.05,
                           ymax=0.95,
                           color='mediumseagreen')

        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(0, len(pred) - 1)
        ax.set_yticks([])
        ax.set_title(title)

    @staticmethod
    def plot_line_vspan_sep(pred,
                            gt,
                            title,
                            ax1: plt.Axes,
                            ax2,
                            ax3,
                            color=(0, 0, 0),
                            thres=0.5):
        x = np.arange(len(gt))
        _len = len(gt)

        # _thres = max(pred.max() * thres, 0.3)
        _thres = max(pred.max() * 0.5, 0.3)
        predt = (pred > _thres).astype(float)

        preds = pred

        # ax1.fill_between(x,
        #                  preds,
        #                  y2=0,
        #                  linewidth=0.50,
        #                  linestyle="-",
        #                  color=color,
        #                  alpha=0.7)
        ax1.plot(x, pred, lw=1.3, color=color)
        for i in range(1, _len - 2):
            ax2.axvspan(xmin=x[i],
                        xmax=x[i + 1],
                        ymin=0,
                        ymax=1,
                        color=color,
                        alpha=predt[i])
            if gt[i] == 1:
                ax3.axvspan(xmin=x[i],
                            xmax=x[i + 1],
                            ymin=0,
                            ymax=1,
                            color='mediumseagreen')

        # ax3.set_ylabel("Ground-truth")
        # ax1.set_ylabel("Prediction Score")
        # ax2.set_ylabel("Prediction Location")
        for ax in [ax1, ax2, ax3]:
            ax.grid(False)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, len(pred) - 1)
            ax.set_yticks([])
            ax.set_xticks([])
            # ax.set_title(title)

    def plot(self, df_gt, dict_pred, shuffle=False, total_images=None, seed=0):
        palette = sns.color_palette(None, len(self.class_dict.keys()))
        video_ids = list(dict_pred.keys())

        df_grp_label = df_gt.groupby('label')

        cont = True
        cnt_ax = 0

        labels = list(df_grp_label.groups.keys())

        images_per_label = (None if total_images is None else int(
            total_images / len(labels)))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(labels)
        for cur_group in tqdm(labels, desc="plotting...", position=5):
            df_cur_group = df_grp_label.get_group(cur_group)
            video_ids = np.unique(df_cur_group['video-id'].values)
            for counter_label_img, vid_id in enumerate(video_ids):

                if len(dict_pred[vid_id]) == 0:
                    continue

                gt_rows = df_cur_group[df_cur_group["video-id"] == vid_id]
                cur_label = gt_rows['label'].values

                assert cur_label[0] == cur_group

                element_logits = dict_pred[vid_id]["logit"]
                if not self.separate:
                    try:
                        ax = self.cur_axes.pop(0)
                    except IndexError:
                        self.cur_fig.tight_layout()
                        self.pdf.savefig(self.cur_fig)
                        plt.close(self.cur_fig)

                        _f, _a = plt.subplots(self.plot_per_page, 1)
                        self.cur_fig = _f
                        self.cur_axes = _a.tolist()
                        ax = self.cur_axes.pop(0)

                        cnt_ax += len(_a)
                        if total_images is not None and cnt_ax >= total_images:
                            cont = False
                            break
                else:
                    fig1, ax1 = plt.subplots(figsize=(6, 0.8))
                    fig2, ax2 = plt.subplots(figsize=(6, 0.8))
                    fig3, ax3 = plt.subplots(figsize=(6, 0.8))

                for cls_idx in np.unique(cur_label):
                    classname = self.class_dict[cls_idx]
                    max_x = dict_pred[vid_id]["duration"]

                    # get gt logit
                    logit_gt = np.zeros(max_x + 1, dtype=np.int)
                    for _, _row in gt_rows[gt_rows['label'] ==
                                           cls_idx].iterrows():
                        ts, te = int(_row['t-start']), int(_row['t-end'])
                        logit_gt[ts:te + 1] = 1

                    logit = element_logits[:, cls_idx]
                    pred = interpolate_logit(logit, max_x + 1)

                    if self.separate:
                        self.plot_line_vspan_sep(pred,
                                                 logit_gt,
                                                 str(vid_id) + " _" +
                                                 str(classname),
                                                 ax1,
                                                 ax2,
                                                 ax3,
                                                 color='royalblue')
                    else:
                        self.plot_line_vspan(pred,
                                             logit_gt,
                                             str(vid_id) + " _" +
                                             str(classname),
                                             ax,
                                             color='royalblue')

                    break

                if self.separate:
                    fig1.savefig(os.path.join(self.folder_name,
                                              str(vid_id) + "_pred.png"),
                                 bbox_inches='tight',
                                 pad_inches=0,
                                 dpi=300)
                    fig2.savefig(os.path.join(self.folder_name,
                                              str(vid_id) + "_thresh.png"),
                                 bbox_inches='tight',
                                 pad_inches=0,
                                 dpi=300)
                    fig3.savefig(os.path.join(self.folder_name,
                                              str(vid_id) + "_gt.png"),
                                 bbox_inches='tight',
                                 pad_inches=0,
                                 dpi=300)
                    plt.close('all')
                else:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(max_x / 15))

                if images_per_label is not None and counter_label_img >= images_per_label:
                    break

            if not cont:
                break

        plt.close("all")
        if not self.separate:
            self.pdf.close()
