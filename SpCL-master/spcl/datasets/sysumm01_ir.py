from __future__ import print_function, absolute_import
import os.path as osp
import re
import random
import numpy as np
from glob import glob

from ..utils.data import BaseImageDataset


class SYSU_MM01_IR(BaseImageDataset):
    dataset_dir = "SYSU-MM01"

    def __init__(self, root='', verbose=True, ncl=1, mode='all', **kwargs):
        super(SYSU_MM01_IR, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'exp/train_id.txt')
        self.val_dir = osp.join(self.dataset_dir, 'exp/val_id.txt')
        self.text_dir = osp.join(self.dataset_dir, 'exp/test_id.txt')

        self._check_before_run()

        self.train_id = self._get_id(self.train_dir) + self._get_id(self.val_dir)
        self.query_id = self._get_id(self.text_dir)
        self.gallery_id = self.query_id

        self.rgb_cams = ['cam1', 'cam2', 'cam4', 'cam5']
        self.ir_cams = ['cam3', 'cam6']
        self.train = self._process_dir(self.train_id, self.ir_cams)
        self.query = self._process_dir(self.query_id, self.ir_cams)
        if mode == 'all':
            # self.gallery = self._process_dir(self.gallery_id, self.rgb_cams)
            self.gallery = self._process_dir_gallery(self.gallery_id, self.rgb_cams)
        elif mode == 'indoor':
            # self.gallery = self._process_dir(self.gallery_id, ['cam1', 'cam2'])
            self.gallery = self._process_dir_gallery(self.gallery_id, ['cam1', 'cam2'])

        if verbose:
            print("=> SYSU-MM01 IR loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.text_dir):
            raise RuntimeError("'{}' is not available".format(self.text_dir))

    def _get_id(self, file_path):
        with open(file_path, 'r') as f:
            ids = f.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]
        return ids

    def _process_dir(self, ids, cams):
        ids_container = list(np.unique(ids))
        id2label = {id_: label for label, id_ in enumerate(ids_container)}

        dataset = []
        for id_ in sorted(ids):
            for cam in cams:
                img_dir = osp.join(self.dataset_dir, cam, id_)
                if osp.isdir(img_dir):
                    img_list = glob(osp.join(img_dir, "*.jpg"))
                    img_list.sort()
                    for img_path in img_list:
                        dataset.append((img_path, id2label[id_], int(cam[-1]) - 1))
        return dataset

    def _process_dir_gallery(self, ids, cams):
        ids_container = list(np.unique(ids))
        id2label = {id_: label for label, id_ in enumerate(ids_container)}

        dataset = []
        for id_ in sorted(ids):
            for cam in cams:
                img_dir = osp.join(self.dataset_dir, cam, id_)
                if osp.isdir(img_dir):
                    img_list = glob(osp.join(img_dir, "*.jpg"))
                    img_list.sort()
                    dataset.append((random.choice(img_list), id2label[id_], int(cam[-1]) - 1))
        return dataset