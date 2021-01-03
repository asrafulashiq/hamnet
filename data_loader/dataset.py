import numpy as np
import glob
import utils
import time
import torch
import os

HOME = os.getcwd()


class Dataset():
    def __init__(self, args, mode='train', sampling='random'):
        self.sampling = sampling
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(
            HOME, 'data/%s-%s-JOINTFeatures.npy' %
            (args.dataset_name, args.feature_type))
        self.path_to_annotations = os.path.join(
            HOME, 'data', self.dataset_name + '-Annotations/')
        try:
            self.features = np.load(self.path_to_features,
                                    encoding='bytes',
                                    allow_pickle=True)
            self.features.shape
        except AttributeError:
            self.features = np.load(self.path_to_features, allow_pickle=True)

        self.segments = np.load(self.path_to_annotations + 'segments.npy',
                                allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy',
                              allow_pickle=True)  # Specific to Thumos14
        self.subset = np.load(self.path_to_annotations + 'subset.npy',
                              allow_pickle=True)
        self.videoname = np.load(self.path_to_annotations + 'videoname.npy',
                                 allow_pickle=True)
        self.duration = np.load(self.path_to_annotations + 'duration.npy',
                                allow_pickle=True)
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.train_subset = 'validation' if 'thumos' in self.dataset_name.lower(
        ) else 'training'

        self.classlist = sorted(
            np.load(self.path_to_annotations + 'classlist.npy',
                    allow_pickle=True))
        activity_index = {
            cls: i
            for i, cls in enumerate([x.decode() for x in self.classlist])
        }
        self.class_dict = {
            i: cls
            for i, cls in enumerate([x.decode() for x in self.classlist])
        }

        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]

        labs_all = np.load(self.path_to_annotations + 'labels.npy',
                           allow_pickle=True)
        self.labels_index = []
        for each in labs_all:
            tmp = [activity_index[_l] for _l in each]
            self.labels_index.append(tmp)

        self.train_test_idx()
        # self.classwise_feature_mapping()
        self.mode = mode
        self.num_segments = args.num_segments

        # annotate indices from train idx
        np.random.seed(args.seed)

        if self.mode == 'train':
            indices = self.trainidx
        else:
            indices = self.testidx
        self.sup_idx = np.random.choice(indices,
                                        size=int(
                                            len(indices) * args.percent_sup),
                                        replace=False)

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == self.train_subset:  # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def __len__(self):
        if self.mode == 'train':
            indices = self.trainidx
        else:
            indices = self.testidx
        return len(indices)

    def __getitem__(self, idx):
        if self.mode == 'train':
            indices = self.trainidx
        else:
            indices = self.testidx

        index = indices[idx]

        feature = self.features[index]
        vid_name = self.videoname[index].decode()
        num_seg = feature.shape[0]
        segments = self.segments[index]
        labs = self.labels_index[index]

        segm = np.zeros((num_seg, self.num_class + 1), dtype=np.float32)
        for ret, _lab in zip(segments, labs):
            ss, ee = ret
            ss, ee = int(ss * 25 / 16), int(ee * 25 / 16)
            segm[ss:ee + 1, _lab] = 1
        segm[:, -1] = 1 - np.max(segm[:, :-1], axis=-1)

        segm = segm if index in self.sup_idx else np.zeros_like(segm)

        label = self.labels_multihot[index]

        if self.sampling == 'pad':
            feature, segm = self.random_pad(feature, segm)
        elif self.sampling == 'avg':
            feature, segm = self.random_avg(feature, segm)
        else:
            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            elif self.sampling == "all":
                sample_idx = np.arange(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')
            feature = feature[sample_idx]
            segm = segm[sample_idx]

        return torch.from_numpy(feature).float(), torch.from_numpy(
            label).float(), torch.from_numpy(segm).float(), vid_name, num_seg

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)
