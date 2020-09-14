import numpy as np
import os


# class Config():
#     def __init__(self, args):
#         for k, v in args.__dict__.items():
#             setattr(self, k, v)

#         self.lr = args.lr
#         self.num_iters = args.num_iters  #len(self.lr)
#         self.num_class = 20
#         self.modal = args.modal
#         if self.modal == 'all':
#             self.len_feature = 2048
#         else:
#             self.len_feature = 1024
#         self.act_thresh = np.arange(0.1, 0.3, 0.05)
#         # self.act_thresh = np.arange(0.1, 0.3, 0.05)
#         self.scale = 1
#         self.gt_path = os.path.join(self.data_path, 'gt.json')
#         self.feature_fps = 25

#         ######
#         self.max_seqlen = self.num_segments
#         self.num_class = 20
#         self.feature_size = self.len_feature
#         self.dataset_name = "Thumos14reduced"
#         self.feature_type = "I3D"

#     def __str__(self):
#         _str = "Configuration: \n"
#         _str += "".join(["-"] * 10)
#         _str += "\n"
#         for k, v in self.__dict__.items():
#             if not k.startswith("__"):
#                 _str += f"{k}: {v}\n"
#         _str += "".join(["-"] * 10)
#         _str += "\n"
#         return _str


class_dict = {
    0: "BaseballPitch",
    1: "BasketballDunk",
    2: "Billiards",
    3: "CleanAndJerk",
    4: "CliffDiving",
    5: "CricketBowling",
    6: "CricketShot",
    7: "Diving",
    8: "FrisbeeCatch",
    9: "GolfSwing",
    10: "HammerThrow",
    11: "HighJump",
    12: "JavelinThrow",
    13: "LongJump",
    14: "PoleVault",
    15: "Shotput",
    16: "SoccerPenalty",
    17: "TennisSwing",
    18: "ThrowDiscus",
    19: "VolleyballSpiking"
}
