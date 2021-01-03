from loguru import logger
import torch
import numpy as np
import utils
from eval.eval_detection import ANETdetection
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

        elem, _, _, element_atn = net(features)

        element_logits = elem * element_atn

        label_np = _label.squeeze().cpu().numpy()

        pred_vid_score = get_cls_score(element_logits, rat=self.config.rat)
        score_np = pred_vid_score.copy()

        self.class_true.append(label_np)
        self.class_pred.append(pred_vid_score)

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

            cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]
            cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

            proposal_dict = {}

            for i in range(len(act_thresh)):
                cas_temp = cas_pred.copy()
                cas_temp_atn = cas_pred_atn.copy()

                seg_list = []

                for c in range(len(pred)):
                    pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
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

        tIoU_thresh = np.array(self.config.tIoU_thresh)
        anet_detection = ANETdetection(
            self.config.gt_path,
            self.final_res,
            subset="test",
            tiou_thresholds=tIoU_thresh,
            verbose=False,
            check_status=False,
        )

        mAP, average_mAP, _ = anet_detection.evaluate()

        for i in range(tIoU_thresh.shape[0]):
            logger.info(f"mAP@{tIoU_thresh[i]:.2f} :: {mAP[i]*100: .2f} %")

        logger.info(f"Average mAP {average_mAP * 100: .2f} %")
        # logger.info(f"\nClass wise scores:\n" + str(details.to_markdown()))

        class_mAP = getClassificationMAP(np.array(self.class_pred),
                                         np.array(self.class_true))
        logger.info(f"Classification mAP {class_mAP:.2f} %")
        logger.info(f"Classification accuracy {test_acc * 100: .2f} %")

        return average_mAP
