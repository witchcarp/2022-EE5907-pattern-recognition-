from dataset import Dataset111
import numpy as np
import scipy.linalg as linalg
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import  matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier


def pca_to2():
    eee = PCA(a=2)
    select_eigvect, data_after_pro = eee.calculate_after(img_train)
    idex1 = np.where(grd_train == 25)
    idex2 = np.where(grd_train != 25)
    for i in range(2):
        plt.imshow(select_eigvect[:, i].reshape(32, 32))
        plt.savefig('pca_to2/eigvect{}.pdf'.format(i))

    plt.figure()
    colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
    for i in range(25):
        plt.scatter(data_after_pro[np.where(grd_train == i), 0], data_after_pro[np.where(grd_train == i), 1], color=next(colors), s=10)
    plt.scatter(data_after_pro[idex1, 0], data_after_pro[idex1, 1], marker='*', s=60, label='self', color='k')
    plt.legend()
    plt.savefig('pca_to2/projected_in_2d.pdf')


def pca_to3():
    fff = PCA(a=3)
    select_eigvect, data_after_pro = fff.calculate_after(img_train)
    idex1 = np.where(grd_train == 25)
    idex2 = np.where(grd_train != 25)
    for i in range(3):
        plt.imshow(select_eigvect[:, i].reshape(32, 32))
        plt.savefig('pca_to3/eigvect{}.pdf'.format(i))
    dpl = plt.figure().add_subplot(projection='3d')
    colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
    for i in range(25):
        dpl.scatter(data_after_pro[np.where(grd_train == i), 0], data_after_pro[np.where(grd_train == i), 1], data_after_pro[np.where(grd_train == i), 2], color=next(colors), s=10)
    dpl.scatter(data_after_pro[idex1, 0], data_after_pro[idex1, 1], data_after_pro[idex1, 2], marker='*', s=60, label='self', color='k')
    plt.legend()
    plt.savefig('pca_to3/projected_in_3d.pdf')


class PCA:
    def __init__(self, a):
        self.a = a

    def calculate_after(self, xx):
        xx_1 = xx - np.mean(xx, axis=0)
        co = np.cov(xx_1.T)
        eigval, eigvect = linalg.eig(co)
        eigvect = np.real(eigvect)
        self.select_eigvect = eigvect[:, :self.a]
        xx_nex = np.dot(xx_1, self.select_eigvect)
        return self.select_eigvect, xx_nex

    def for_testset(self, xx):
        xx_1 = xx - np.mean(xx, axis=0)
        xx_nex = np.dot(xx_1, self.select_eigvect)
        return xx_nex


def knn_clas(i_train, g_train, i_test, g_test):

    idex1 = np.where(g_test == 25)
    idex2 = np.where(g_test != 25)
    g_test_PIE = g_test[idex2]
    g_test_self = g_test[idex1]

    for dim in [40, 80, 200]:
        pca_ml = PCA(a=dim)
        _, pca_x_train = pca_ml.calculate_after(i_train)
        pca_x_test = pca_ml.for_testset(i_test)
        knn_ml = KNeighborsClassifier(n_neighbors=1)
        knn_ml.fit(pca_x_train, g_train)

        y_pred = knn_ml.predict(pca_x_test)
        acc_pie = accuracy_score(y_pred[idex2], g_test_PIE)
        print('for dim {}: PIE accuracy {}'.format(dim, acc_pie))
        acc_self = accuracy_score(y_pred[idex1], g_test_self)
        print('for dim {}: self accuracy {}'.format(dim, acc_self))


if __name__ == '__main__':
    data1 = Dataset111('./PIE/')
    img_tr, grd_tr = data1.load_data()
    img_test, grd_test = data1.load_data(train=False)
    img_tr = img_tr.reshape(img_tr.shape[0], -1)
    img_test = img_test.reshape(img_test.shape[0], -1)
    index = list(range(img_tr.shape[0]))
    np.random.shuffle(index)
    img_train = img_tr[index[:500]]
    grd_train = grd_tr[index[:500]]
    # pca_to2()
    # pca_to3()
    knn_clas(img_tr, grd_tr, img_test, grd_test)









