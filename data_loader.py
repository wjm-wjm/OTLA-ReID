import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import collections


def mask_outlier(pseudo_labels):
    """
    Mask outlier data of clustering results.
    """
    index2label = collections.defaultdict(int)
    for label in pseudo_labels:
        index2label[label.item()] += 1
    nums = np.fromiter(index2label.values(), dtype=float)
    labels = np.fromiter(index2label.keys(), dtype=float)
    train_labels = labels[nums > 1]

    return np.array([i in train_labels for i in pseudo_labels])


def read_image(data_files, pid2label, img_w, img_h):
    train_img = []
    train_label = []
    for img_path in data_files:
        # img
        img = Image.open(img_path)
        img = img.resize((img_w, img_h), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)

    return np.array(train_img), np.array(train_label)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def pre_process_sysu(args, data_dir):
    rgb_cameras = ["cam1", "cam2", "cam4", "cam5"]
    ir_cameras = ["cam3", "cam6"]

    # load id info
    file_path_train = os.path.join(data_dir, "exp/train_id.txt")
    file_path_val = os.path.join(data_dir, "exp/val_id.txt")
    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_train = ["%04d" % x for x in ids]

    with open(file_path_val, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_val = ["%04d" % x for x in ids]

    # combine train and val split
    id_train.extend(id_val)

    files_rgb = []
    files_ir = []
    for id in sorted(id_train):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    # relabel
    pid_container = set()
    for img_path in files_ir:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    train_color_image, train_color_label = read_image(files_rgb, pid2label, args.img_w, args.img_h)
    # np.save(os.path.join(data_dir, 'train_rgb_resized_img.npy'), train_color_image)
    # np.save(os.path.join(data_dir, 'train_rgb_resized_label.npy'), train_color_label)
    train_thermal_image, train_thermal_label = read_image(files_ir, pid2label, args.img_w, args.img_h)
    # np.save(os.path.join(data_dir, 'train_ir_resized_img.npy'), train_thermal_image)
    # np.save(os.path.join(data_dir, 'train_ir_resized_label.npy'), train_thermal_label)

    return train_color_image, train_color_label, train_thermal_image, train_thermal_label


def pre_process_regdb(args, data_dir):
    train_color_list = os.path.join(data_dir, "idx/train_visible_{}".format(args.trial) + ".txt")
    train_thermal_list = os.path.join(data_dir, "idx/train_thermal_{}".format(args.trial) + ".txt")

    color_img_file, train_color_label = load_data(train_color_list)
    thermal_img_file, train_thermal_label = load_data(train_thermal_list)

    train_color_image = []
    for i in range(len(color_img_file)):
        img = Image.open(data_dir + color_img_file[i])
        img = img.resize((args.img_w, args.img_h), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_color_image.append(pix_array)
    train_color_image = np.array(train_color_image)
    train_color_label = np.array(train_color_label)

    train_thermal_image = []
    for i in range(len(thermal_img_file)):
        img = Image.open(data_dir + thermal_img_file[i])
        img = img.resize((args.img_w, args.img_h), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_thermal_image.append(pix_array)
    train_thermal_image = np.array(train_thermal_image)
    train_thermal_label = np.array(train_thermal_label)

    return train_color_image, train_color_label, train_thermal_image, train_thermal_label


class SYSUData(data.Dataset):
    def __init__(self, args, data_dir, transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images and labels
        self.train_color_image, self.train_color_label, self.train_thermal_image, self.train_thermal_label = pre_process_sysu(args, data_dir)

        if args.setting == "unsupervised":
            self.train_color_image = np.load(os.path.join(data_dir, args.train_visible_image_path))
            self.train_color_label = np.load(os.path.join(data_dir, args.train_visible_label_path))

            mask = mask_outlier(self.train_color_label)
            self.train_color_image = self.train_color_image[mask]
            self.train_color_label = self.train_color_label[mask]
            ids_container = list(np.unique(self.train_color_label))
            id2label = {id_: label for label, id_ in enumerate(ids_container)}
            for i, label in enumerate(self.train_color_label):
                self.train_color_label[i] = id2label[label]

        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.args = args

    def __getitem__(self, index):
        img1, label1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, label2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform_train_rgb(img1)
        img2 = self.transform_train_ir(img2)

        return img1, img2, label1, label2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, args, data_dir, transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images and labels
        self.train_color_image, self.train_color_label, self.train_thermal_image, self.train_thermal_label = pre_process_regdb(args, data_dir)

        if args.setting == "unsupervised":
            self.train_color_image = np.load(os.path.join(data_dir, args.train_visible_image_path))
            self.train_color_label = np.load(os.path.join(data_dir, args.train_visible_label_path))

            mask = mask_outlier(self.train_color_label)
            self.train_color_image = self.train_color_image[mask]
            self.train_color_label = self.train_color_label[mask]
            ids_container = list(np.unique(self.train_color_label))
            id2label = {id_: label for label, id_ in enumerate(ids_container)}
            for i, label in enumerate(self.train_color_label):
                self.train_color_label[i] = id2label[label]

        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.args = args

    def __getitem__(self, index):
        img1, label1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2, label2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform_train_rgb(img1)
        img2 = self.transform_train_ir(img2)

        return img1, img2, label1, label2

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform_test=None, img_size=None):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform_test = transform_test

    def __getitem__(self, index):
        img1, label1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform_test(img1)

        return img1, label1

    def __len__(self):
        return len(self.test_image)
