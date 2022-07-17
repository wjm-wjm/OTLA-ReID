import os
import sys
import random
import copy
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler


def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos


def GenIdx_single(label):
    pos = []
    num = []
    unique_label = np.unique(label)
    for i in range(np.max(unique_label)+1):
        if i in unique_label:
            tmp_pos = [k for k, v in enumerate(label) if v == i]
            pos.append(tmp_pos)
            num.append(len(tmp_pos))
        else:
            pos.append([])
            num.append(0)

    return pos, np.array(num) / np.array(num).sum()

    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)

    return sample_pos


class IdentitySampler(Sampler):
    """
    Note: color = rgb = visible, thermal = ir = infrared.
    This is the data sampler code with real supervision of both modalities.
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, dataset_num_size):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = dataset_num_size * np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class SemiIdentitySampler_randomIR(Sampler):
    """
    Note: color = rgb = visible, thermal = ir = infrared.
    This is the data sampler code with rgb real (pseudo) supervision
    but without ir pseudo supervision (randomly selected for infrared modality).
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, num_pos, batchSize, dataset_num_size):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = dataset_num_size * np.maximum(len(train_color_label), len(train_thermal_label))

        batch_idx_list = []
        uni_label_temp = uni_label
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = []
            for i in range(batchSize):
                if len(uni_label_temp) == 0:
                    uni_label_temp = uni_label
                idx = random.randint(0, len(uni_label_temp)-1)
                batch_idx.append(uni_label_temp[idx])
                uni_label_temp = np.delete(uni_label_temp, idx)
            batch_idx_list.append(np.array(batch_idx))

        thermal_pos_num = list(range(train_thermal_label.size))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = batch_idx_list[j]
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = []
                for k in range(num_pos):
                    if len(thermal_pos_num) == 0:
                        thermal_pos_num = list(range(train_thermal_label.size))
                    idx = random.randint(0, len(thermal_pos_num) - 1)
                    sample_thermal.append(thermal_pos_num[idx])
                    thermal_pos_num.pop(idx)
                sample_thermal = np.array(sample_thermal)

                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class SemiIdentitySampler_pseudoIR(Sampler):
    """
    Note: color = rgb = visible, thermal = ir = infrared.
    This is the data sampler code with ir pseudo supervision and rgb real (pseudo) supervision.
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, num_pos, batchSize, dataset_num_size):
        uni_label_color = np.unique(train_color_label)
        self.n_classes_color = len(uni_label_color)
        uni_label_thermal = np.unique(train_thermal_label)
        self.n_classes_thermal = len(uni_label_thermal)
        thermal_pos, _ = GenIdx_single(train_thermal_label)
        print(uni_label_thermal.size)

        N = dataset_num_size * np.maximum(len(train_color_label), len(train_thermal_label))

        batch_idx_list = []
        uni_label_temp = uni_label_thermal
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = []
            for i in range(batchSize):
                if len(uni_label_temp) == 0:
                    uni_label_temp = uni_label_thermal
                idx = random.randint(0, len(uni_label_temp) - 1)
                batch_idx.append(uni_label_temp[idx])
                uni_label_temp = np.delete(uni_label_temp, idx)
            batch_idx_list.append(np.array(batch_idx))

        color_pos_temp = copy.deepcopy(color_pos)
        thermal_pos_temp = copy.deepcopy(thermal_pos)
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = batch_idx_list[j]
            for i in range(batchSize):
                sample_color = []
                sample_thermal = []
                for k in range(num_pos):
                    if len(color_pos_temp[batch_idx[i]]) == 0:
                        color_pos_temp[batch_idx[i]] = copy.deepcopy(color_pos[batch_idx[i]])
                    if len(thermal_pos_temp[batch_idx[i]]) == 0:
                        thermal_pos_temp[batch_idx[i]] = copy.deepcopy(thermal_pos[batch_idx[i]])
                    idx_c = random.randint(0, len(color_pos_temp[batch_idx[i]]) - 1)
                    idx_t = random.randint(0, len(thermal_pos_temp[batch_idx[i]]) - 1)
                    sample_color.append(color_pos_temp[batch_idx[i]][idx_c])
                    sample_thermal.append(thermal_pos_temp[batch_idx[i]][idx_t])
                    color_pos_temp[batch_idx[i]].pop(idx_c)
                    thermal_pos_temp[batch_idx[i]].pop(idx_t)
                sample_color = np.array(sample_color)
                sample_thermal = np.array(sample_thermal)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class AverageMeter(object):
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def sort_list_with_unique_index(initial_list):
    """
    Returns the first and last index of each unique value.
    Return:
        s1[idx_]: first index of each unique value
        s2[idx_]: last index of each unique value
        num[idx_]: total number of items of each unique value
    """
    a = np.asarray(initial_list)
    a_u, idx = np.unique(a, return_index=True)
    idx_ = a[np.sort(idx)]
    s1 = np.ones(a_u[-1]+1, dtype=int) * -1
    s2 = np.ones(a_u[-1]+1, dtype=int) * -1
    num = np.zeros(a_u[-1]+1, dtype=int)
    s3 = defaultdict(list)
    for i, a_v in enumerate(a):
        if (a_v in a_u) and (s1[a_v] == -1):
            s1[a_v] = i
            s2[a_v] = i
            num[a_v] = 1
            s3[a_v].append(i)
        elif (a_v in a_u) and (s1[a_v] != -1):
            s2[a_v] = i
            num[a_v] += 1
            s3[a_v].append(i)

    return s1[idx_], s2[idx_], num[idx_], idx_, s3


