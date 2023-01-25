import numpy as np
import os
import cv2


class Dataset111():
    def __init__(self, dataroot):
        self.num = 25
        self.train_ratio = 0.7
        self.dataroot = dataroot
        self._select = self.select_from_all()
        self.trainlist, self.testlist = self.train_test()
        self.labelassign = {label: i for i, label in enumerate(self._select)}

    def select_from_all(self):
        _all = list(range(1, 69))
        np.random.shuffle(_all)
        _select = _all[:self.num]
        _select.append('selff')
        print(_select)
        return _select

    def get_subset(self, rooot):
        samples = os.listdir(rooot)
        return samples, len(samples)

    def train_test(self):
        train_list = []
        test_list = []
        data_list = []
        num = 0
        for a in self._select:
            selet_data_list, numm = self.get_subset(os.path.join('./PIE', str(a)))
            train_list.extend((os.path.join('./PIE', str(a), x), a)
                              for x in selet_data_list[:int(self.train_ratio*len(selet_data_list))])
            test_list.extend((os.path.join('./PIE', str(a), x), a)
                             for x in selet_data_list[int(self.train_ratio * len(selet_data_list)):])

        return train_list, test_list

    def load_data(self, train=True):
        imgess = []
        groun = []
        if train:
            flist = self.trainlist
        else:
            flist = self.testlist

        for path, label in flist:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gro = self.labelassign[label]
            imgess.append(img)
            groun.append(gro)
        return np.stack(imgess, axis=0)/255, np.stack(groun)


if __name__ == '__main__':

    data1 = Dataset111('./PIE/')
    img_train, grd_train = data1.load_data()

