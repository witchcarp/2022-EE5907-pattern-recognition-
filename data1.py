import os
import numpy as np
import json
import cv2
from numpy.core.fromnumeric import resize
from torch.utils.data import DataLoader, Dataset
import torch


class PIEDataSet1(Dataset):
    num_subject = 25
    ratio_train = 0.7
    num_all_subject = 68

    def __init__(self, data_root, if_train):
        self.if_train = if_train
        self.data_root = data_root
        self.subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 'selff']
        self.train_list, self.test_list = self.train_test_split()
        print(f'Num subject {len(self.subject)}, Num train {len(self.train_list)}, Num Test {len(self.test_list)}')
        self.label2gt = {c: i for i, c in enumerate(self.subject)}
        imgs = []
        gts = []

        if self.if_train == True:
            datalist = self.train_list
        else:
            datalist = self.test_list
        for path, label in datalist:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gt_scalar = self.label2gt[label]
            a = torch.zeros((26))
            a[gt_scalar] = 1
            imgs.append(img)
            gts.append(a)
        self.im = np.stack(imgs) / 255
        self.gt = np.stack(gts)


    def train_test_split(self):
        print('Spliting Training and Testing')
        train_list = []
        test_list = []
        for s in self.subject:
            subject_path = os.path.join(self.data_root, str(s))
            all_samples = os.listdir(subject_path)
            train_num = int(len(all_samples) * self.ratio_train)
            np.random.shuffle(all_samples)
            train_list.extend([(os.path.join(subject_path, sample), s)
                               for sample in all_samples[:train_num]])
            test_list.extend([(os.path.join(subject_path, sample), s)
                              for sample in all_samples[train_num:]])

        return train_list, test_list

    def __len__(self):
        return len(self.im)

    def __getitem__(self, index):
        return self.im[index], self.gt[index, :]


if __name__ == '__main__':
    a = PIEDataSet1('./PIE/', if_train=True)
    print(a.test_list)

