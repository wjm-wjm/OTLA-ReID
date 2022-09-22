import os
import errno
from PIL import Image
import numpy as np
import collections
import torch


def mkdir_if_missing(dir_path):
    """
    Create file if missing.
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint_pseudo_label(state, fpath="checkpoint.pth.tar"):
    """
    Save model for generating pseudo label.
    """
    mkdir_if_missing(os.path.dirname(fpath))
    torch.save(state, fpath)


def mask_outlier(train_pseudo_label):
    """
    Mask outlier data of clustering results.
    """
    index2label = collections.defaultdict(int)
    for label in train_pseudo_label:
        index2label[label.item()] += 1
    nums = np.fromiter(index2label.values(), dtype=float)
    label = np.fromiter(index2label.keys(), dtype=float)
    train_label = label[nums > 1]

    return np.array([i in train_label for i in train_pseudo_label])


def R_gt(train_real_label, train_pseudo_label):
    '''
    The Average Maximum Proportion of Ground-truth Classes (R_gt) in supplementary material.
    '''
    p = 0
    mask = mask_outlier(train_pseudo_label)
    train_real_label = train_real_label[mask]
    ids_container = list(np.unique(train_real_label))
    id2label = {id_: label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(train_real_label):
        train_real_label[i] = id2label[label]
    train_pseudo_label = train_pseudo_label[mask]
    ids_container = list(np.unique(train_pseudo_label))
    id2label = {id_: label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(train_pseudo_label):
        train_pseudo_label[i] = id2label[label]
    for i in range(np.unique(train_real_label).size):
        sample_id = (train_real_label == i)
        sample_label = train_pseudo_label[sample_id]
        sample_num_per_label = np.zeros(np.unique(train_pseudo_label).size)
        for j in sample_label:
            sample_num_per_label[j] += 1
        p_i = np.max(sample_num_per_label) / sample_label.size
        p += p_i
    p = p / np.unique(train_real_label).size
    print("R_gt: {:.4f}".format(p))

    return p


def R_ct(train_real_label, train_pseudo_label):
    '''
    The Average Maximum Proportion of Pseudo Classes (R_ct) in supplementary material.
    '''
    p = 0
    mask = mask_outlier(train_pseudo_label)
    train_real_label = train_real_label[mask]
    ids_container = list(np.unique(train_real_label))
    id2label = {id_: label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(train_real_label):
        train_real_label[i] = id2label[label]
    train_pseudo_label = train_pseudo_label[mask]
    ids_container = list(np.unique(train_pseudo_label))
    id2label = {id_: label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(train_pseudo_label):
        train_pseudo_label[i] = id2label[label]
    for i in range(np.unique(train_pseudo_label).size):
        sample_id = (train_pseudo_label == i)
        sample_label = train_real_label[sample_id]
        sample_num_per_label = np.zeros(np.unique(train_real_label).size)
        for j in sample_label:
            sample_num_per_label[j] += 1
        p_i = np.max(sample_num_per_label) / sample_label.size
        p += p_i
    p = p / np.unique(train_pseudo_label).size
    print("R_ct: {:.4f}".format(p))

    return p


def P_v(train_real_label, train_pseudo_label):
    '''
    The Proportion of Visible Training Samples (P_v).
    '''
    len_data = len(train_real_label)
    mask = mask_outlier(train_pseudo_label)
    len_mask_data = len(train_pseudo_label[mask])
    p = len_mask_data / len_data
    print("P_v: {:.4f}, total samples: {}, total samples without outliers: {}".format(p, len_data, len_mask_data))

    return p


def Q_v(train_real_label, train_pseudo_label):
    mask = mask_outlier(train_pseudo_label)
    n_class = np.unique(train_real_label).size
    n_cluster_class = np.unique(train_pseudo_label[mask]).size
    p = np.min((n_class, n_cluster_class)) / np.max((n_class, n_cluster_class))
    print("Q_v: {:.4f}, number of real classes: {}, number of pseudo classes: {}".format(p, n_class, n_cluster_class))

    return p


def R_plq(train_real_label, train_pseudo_label):
    '''
    The Final Metric (R_plq) in supplementary material.
    '''
    R_gt_p = R_gt(train_real_label, train_pseudo_label)
    R_ct_p = R_ct(train_real_label, train_pseudo_label)
    P_v_p = P_v(train_real_label, train_pseudo_label)
    Q_v_p = Q_v(train_real_label, train_pseudo_label)
    R_plq_p = (R_gt_p + R_ct_p) / 2 * P_v_p * Q_v_p
    print("R_plq: {:.4f}".format(R_plq_p))

    return R_gt_p, R_ct_p, P_v_p, Q_v_p, R_plq_p


def save_image_label(train_image_path, train_pseudo_label, train_real_label, model, epoch, logs_dir, save_path,
                     img_size=(144, 288), source_domain="market1501", target_domain="sysumm01_rgb", method_name="spcl_uda"):
    train_image = []
    for fname in train_image_path:
        img = Image.open(fname)
        img = img.resize(img_size, Image.ANTIALIAS)
        pix_array = np.array(img)
        train_image.append(pix_array)

    train_image = np.array(train_image)
    train_pseudo_label = np.array(train_pseudo_label)
    train_real_label = np.array(train_real_label)

    ids_container = list(np.unique(train_pseudo_label))
    id2label = {id_: label for label, id_ in enumerate(ids_container)}
    for i, label in enumerate(train_pseudo_label):
        train_pseudo_label[i] = id2label[label]

    R_gt_p, R_ct_p, P_v_p, Q_v_p, R_plq_p = R_plq(train_real_label, train_pseudo_label)

    np.save(os.path.join(save_path, method_name+"_"+source_domain+"TO"+target_domain+"_"+"train_rgb_resized_img.npy"), train_image)
    np.save(os.path.join(save_path, method_name+"_"+source_domain+"TO"+target_domain+"_"+"train_rgb_resized_label.npy"), train_pseudo_label)

    save_checkpoint_pseudo_label({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "R_gt": R_gt_p,
                "R_ct": R_ct_p,
                "P_v": P_v_p,
                "Q_v": Q_v_p,
                "R_plq": R_plq_p,
            }, fpath=os.path.join(logs_dir, "checkpoint_pseudo_label.pth.tar"))