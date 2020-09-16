from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import os
import json
from eval.eval_detection import ANETdetection
from tqdm import tqdm
from collections import defaultdict
from eval.utils_eval import getClassificationMAP


def get_cls_score(element_cls, dim=-2, rat=20, ind=None):

    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] // rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score


class Tester():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_correct = 0
        self.num_total = 0
        self.dict_pred = defaultdict(dict)

        self.class_true = []
        self.class_pred = []

        final_res = {}
        final_res["results"] = {}
        self.final_res = final_res

    @torch.no_grad()
    def eval_one_batch(self, data, net, class_dict):
        features, _label, segm, vid_name, vid_num_seg = data

        if _label.sum() == 0:
            return

        # if self.config.lm_2 == 0:
        #     element_logits, _, _, element_atn = net(features)
        # else:
        elem, _, _, element_atn = net(features)

        # _min = elem.min(dim=-2, keepdim=True)[0]
        # element_logits = (elem - _min) * element_atn + _min
        element_logits = elem * element_atn

        label_np = _label.squeeze().cpu().numpy()

        pred_vid_score = get_cls_score(element_logits, rat=self.config.rat)
        score_np = pred_vid_score.copy()

        self.class_true.append(label_np)
        self.class_pred.append(pred_vid_score)

        # self.config.class_thresh = pred_vid_score.max() * 0.9
        score_np[score_np < self.config.class_thresh] = 0
        score_np[score_np >= self.config.class_thresh] = 1

        correct_pred = np.sum(label_np == score_np)

        self.num_correct += np.sum(
            (correct_pred == self.config.num_class).astype(np.float32))
        self.num_total += 1

        cas_supp = element_logits[..., :-1]
        cas_supp_atn = element_atn

        logit_atn = cas_supp_atn.expand_as(
            cas_supp).squeeze().data.cpu().numpy()

        self.dict_pred[vid_name[0]]["logit"] = logit_atn
        self.dict_pred[vid_name[0]]["duration"] = int(vid_num_seg.item() * 16 /
                                                      25)

        pred = np.where(pred_vid_score >= self.config.class_thresh)[0]

        # NOTE: threshold
        act_thresh = self.config.act_thresh

        if len(pred) > 0:
            cas_pred = cas_supp[0].cpu().numpy()[:, pred]
            num_segments = cas_pred.shape[0]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
            cas_pred = utils.upgrade_resolution(cas_pred, self.config.scale)

            cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]
            cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))
            cas_pred_atn = utils.upgrade_resolution(cas_pred_atn,
                                                    self.config.scale)

            proposal_dict = {}

            for i in range(len(act_thresh)):
                cas_temp = cas_pred.copy()
                cas_temp_atn = cas_pred_atn.copy()

                cas_norm = (cas_temp - cas_temp.min(axis=0, keepdims=True)) / (
                    cas_temp.max(axis=0, keepdims=True) -
                    cas_temp.min(axis=0, keepdims=True) + 1e-4)

                seg_list = []

                for c in range(len(pred)):
                    pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
                    # pos = np.where(cas_norm[:, c, 0] > act_thresh[i])
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list,
                                                   cas_temp,
                                                   pred_vid_score,
                                                   pred,
                                                   self.config.scale,
                                                   vid_num_seg[0].cpu().item(),
                                                   self.config.feature_fps,
                                                   num_segments,
                                                   gamma=self.config.gamma_oic)

                for j in range(len(proposals)):
                    try:
                        class_id = proposals[j][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[j]
                    except IndexError:
                        logger.error(f"Index error")

            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(
                    utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
            self.final_res["results"][vid_name[0]] = utils.result2json(
                final_proposals, class_dict)

    def final(self, logger=None, class_dict=None):
        if logger is None:
            import loguru
            logger = loguru.logger

        test_acc = self.num_correct / self.num_total

        # tIoU_thresh = np.linspace(0.1, 0.9, 9)
        tIoU_thresh = np.array(self.config.tIoU_thresh)
        anet_detection = ANETdetection(
            self.config.gt_path,
            self.final_res,
            subset="test",
            tiou_thresholds=tIoU_thresh,
            verbose=False,
            check_status=False,
        )

        mAP, average_mAP, details = anet_detection.evaluate()

        for i in range(tIoU_thresh.shape[0]):
            logger.info(f"mAP@{tIoU_thresh[i]:.2f} :: {mAP[i]*100: .2f} %")

        logger.info(f"Average mAP {average_mAP * 100: .2f} %")
        # logger.info(f"\nClass wise scores:\n" + str(details.to_markdown()))

        class_mAP = getClassificationMAP(np.array(self.class_pred),
                                         np.array(self.class_true))
        logger.info(f"Classification mAP {class_mAP:.2f} %")
        logger.info(f"Classification accuracy {test_acc * 100: .2f} %")

        if self.config.save_pred is not None:
            # save prediction to a file
            data_save = {}
            data_save["df_loc"] = details
            data_save["class_mAP"] = class_mAP
            data_save["class_accuracy"] = test_acc
            data_save["df_gt"] = anet_detection.ground_truth
            data_save["df_pred"] = self.dict_pred
            np.save(str(self.config.save_pred), data_save)

        # For debug
        if __debug__:
            gt = anet_detection.ground_truth
            pr = anet_detection.prediction

            def fn(x, n):
                return x[x['video-id'] == n].sort_values('t-start')

            pass

        if self.config.plot:
            from utils import PlotCam

            filepath = os.path.join(self.config.output_path,
                                    self.config.model_name + ".pdf")
            plotter = PlotCam(
                filepath,
                class_dict=class_dict,
            )
            plotter.plot(anet_detection.ground_truth,
                         self.dict_pred,
                         total_images=self.config.plot_total_images,
                         shuffle=False,
                         seed=self.config.seed)

        return average_mAP
