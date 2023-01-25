from dataset import Dataset111
import numpy as np
import scipy.linalg as linalg
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.cm as cm


def lda_to2():
    eee = LDA(a=2)
    select_eigvect, data_after_pro = eee.calculate_after(img_tr, grd_tr)
    idex1 = np.where(grd_tr == 25)
    idex2 = np.where(grd_tr != 25)
    for i in range(2):
        plt.imshow(select_eigvect[:, i].reshape(32, 32))
        plt.savefig('lda_to2/eigvect{}.pdf'.format(i))
    plt.figure()
    colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
    for i in range(25):
        plt.scatter(data_after_pro[np.where(grd_tr == i), 0], data_after_pro[np.where(grd_tr == i), 1], color=next(colors), s=10)
    plt.scatter(data_after_pro[idex1, 0], data_after_pro[idex1, 1], marker='*', label='self',color='k', s=60)
    plt.legend()
    plt.savefig('lda_to2/projected_in_2d.pdf')


def lda_to3():
    fff = LDA(a=3)
    select_eigvect, data_after_pro = fff.calculate_after(img_tr, grd_tr)
    idex1 = np.where(grd_tr == 25)
    for i in range(3):
        plt.imshow(select_eigvect[:, i].reshape(32, 32))
        plt.savefig('lda_to3/eigvect{}.pdf'.format(i))
    dpl = plt.figure().add_subplot(projection='3d')
    colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
    for i in range(25):
        dpl.scatter(data_after_pro[np.where(grd_tr == i), 0], data_after_pro[np.where(grd_tr == i), 1], data_after_pro[np.where(grd_tr == i), 2], color=next(colors), s=10)
    dpl.scatter(data_after_pro[idex1, 0], data_after_pro[idex1, 1], data_after_pro[idex1, 2], marker='*', label='self', color='k', s=60)
    plt.legend()
    plt.savefig('lda_to3/projected_in_3d.pdf')


class LDA:
    def __init__(self, a):
        self.a = a

    def calculate_after(self, xx, yy):
        sw = np.zeros((xx.shape[1], xx.shape[1]))
        sb = np.zeros((xx.shape[1], xx.shape[1]))
        kinds = np.unique(yy)
        mean_all = np.mean(xx, axis=0)
        for i in kinds:
            xx_i = xx[yy==i]
            ni = xx_i.shape[0]
            xx_i_mean = np.mean(xx_i, axis=0)
            sbi = np.outer((xx_i_mean - mean_all), (xx_i_mean - mean_all))
            sb = sb + ni*sbi
            xx_i1 = xx_i - xx_i_mean
            swi = np.dot(xx_i1.T, xx_i1)
            sw = sw + swi

        eigvals, eigvects = np.linalg.eig(np.dot(np.linalg.inv(sw), sb))
        eigvects = np.real(eigvects)
        self.select_eigvect = eigvects[:, :self.a]
        xx_nex = np.dot(xx, self.select_eigvect)
        return self.select_eigvect, xx_nex

    def for_testset(self, xx):
        xx_nex = np.dot(xx, self.select_eigvect)
        return xx_nex

def knn_clas(i_train, g_train, i_test, g_test):

    idex1 = np.where(g_test == 25)
    idex2 = np.where(g_test != 25)

    g_test_PIE = g_test[idex2]
    g_test_self = g_test[idex1]

    for dim in [2, 3, 9]:
        pca_ml = LDA(a=dim)
        _, pca_x_train = pca_ml.calculate_after(i_train, g_train)
        pca_x_test = pca_ml.for_testset(i_test)

        if dim==2:
            plt.scatter(pca_x_test[idex1, 0], pca_x_test[idex1, 1], marker='*', label='self', color='k', s=60)
            plt.scatter(pca_x_test[idex2, 0], pca_x_test[idex2, 1])
            plt.show()



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
    # lda_to2()
    # lda_to3()
    knn_clas(img_tr, grd_tr, img_test, grd_test)



