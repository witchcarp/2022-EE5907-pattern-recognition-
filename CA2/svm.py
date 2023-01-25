
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataset import Dataset111


def SVMM(select_type, train_x, test_x, train_y, test_y):
    # raw vectorized face images
    if select_type == 0:
        x_train_after = train_x
        x_test_after = test_x
    else:
        n_pca = PCA(n_components=select_type)
        n_pca.fit(train_x)
        x_train_after = n_pca.transform(train_x)
        x_test_after = n_pca.transform(test_x)
    result = []
    for penalty in [0.01, 0.1, 1]:
        svm = SVC(C=penalty, kernel='linear')
        svm.fit(x_train_after, train_y)
        predictions = svm.predict(x_test_after)
        score = accuracy_score(test_y, predictions)
        result.append(score)
        if select_type == 0:
            print('Accuracy with penalty({}) for raw images: {}'.format(penalty, score))
        else:
            print('Accuracywith penalty({}) at {} dimensions: {}'.format(penalty, select_type, score))
    return result



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

    fig = plt.figure()
    result_raw = SVMM(0, img_train, img_test, grd_train, grd_test)
    result_80 = SVMM(80, img_train, img_test, grd_train, grd_test)
    result_200 = SVMM(200, img_train, img_test, grd_train, grd_test)
    result = np.stack((result_raw, result_80, result_200))
    labels = [0.01, 0.1, 1]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, result_200, width / 3, label='200', color='lightpink')
    plt.bar(x, result_80, width / 3, label='80', color='skyblue')
    plt.bar(x + width / 2, result_raw, width / 3, label='raw', color='palegreen')

    plt.xticks([0, 1, 2], labels)
    plt.xlabel('penalty')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    for i in x:
        plt.text(x=i - width /2, y=result_200[i] + 0.01, s=round(result_200[i], 3))
        plt.text(x=i, y=result_80[i] + 0.01, s=round(result_80[i], 3))
        plt.text(x=i + width / 2, y=result_raw[i] + 0.01, s=round(result_raw[i], 3))
    plt.show()
