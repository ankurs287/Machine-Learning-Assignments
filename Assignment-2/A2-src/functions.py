import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC


class MySVM:

    def __init__(self, kernel=None, gamma=None, C=1):
        self.clf = SVC(kernel='linear', C=C)
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        kx = X.copy()
        if self.kernel is not None:
            kX = np.hstack((X, self.kernel(X, self.gamma)))
        w = self.clf._get_coef()
        b = self.clf.intercept_
        return predict(kX, w, b)


def predict(X, w, b):
    r = np.dot(X, w.T) + b
    r = np.where(r >= 0, 1, 0)
    return r


def confusion_matrix(y_true, y_pred):
    tp = np.sum(((y_true == 1) & (y_pred == 1)).astype(int))
    tn = np.sum(((y_true == 0) & (y_pred == 0)).astype(int))
    fp = np.sum(((y_true == 0) & (y_pred == 1)).astype(int))
    fn = np.sum(((y_true == 1) & (y_pred == 0)).astype(int))
    cm = np.array([[tn, fp], [fn, tp]])
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    return sn


def roc_plot(y_test, preds):
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


def quad(X, Y):
    return np.square(np.dot(X, Y.T))


def accuracy(y_pred, y_true):
    num = np.sum(y_pred == y_true)
    return num / len(y_true)


def poly_kernel(X, gamma=1):
    return np.square(X)


def rbf_kernel(X, gamma):
    return np.exp(gamma * np.square(X))


def plot_decision_boudary(clf, X, y, kernel=None, gamma=1):
    w = clf._get_coef()[0]
    b = clf.intercept_

    pad = 0.25
    h = 1000
    x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, h), np.linspace(x2_min, x2_max, h))

    plt.scatter(X[:, 0], X[:, 1], c=y)

    # if kernel is not None:
    #     F = w[0] * x1 + w[1] * x2 + w[2] * kernel(x1, gamma) + w[3] * kernel(x2, gamma) + b
    # else:
    #     F = w[0] * x1 + w[1] * x2 + b
    # plt.contour(x1, x2, F, [0], colors='red')

    # start custom
    if kernel is not None:
        Z = predict(np.c_[x1.ravel(), x2.ravel(), kernel(x1.ravel(), gamma), kernel(x2.ravel(), gamma)], w, b)
    else:
        Z = predict(np.c_[x1.ravel(), x2.ravel()], w, b)
    Z = Z.reshape(x1.shape)
    plt.contourf(x1, x2, Z, alpha=0.2)
    # end custom

    return plt


def f1_score(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return np.round(2 * p * r / (p + r), 2)


def one_vs_all(X, y):
    models = []
    for y_i in range(3):
        y_ = y.copy()
        y_ = (y_ == y_i).astype(int)

        clf = SVC(kernel='linear')
        clf.fit(X, y_)

        models.append(clf)
    return models


def one_vs_one(X, y, kernel=None, gamma=1):
    models = []
    kX = X.copy()
    if kernel is not None:
        kX = np.hstack((X, kernel(X, gamma)))
    for y_i in range(2):
        for y_j in range(y_i + 1, 3):
            e1 = np.where(y == y_i)[0]
            e2 = np.where(y == y_j)[0]
            y_ = np.hstack((y[e1], y[e2]))
            X_ = np.vstack((kX[e1], kX[e2]))
            p = np.random.permutation(len(X_))
            y_ = y_[p]
            X_ = X_[p]

            clf = SVC(kernel='linear')
            clf.fit(X_, y_)

            models.append(clf)
    return models
