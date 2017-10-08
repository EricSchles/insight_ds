# -*- coding: utf-8 -*-
# @Author: C. Marcus Chuang

from __future__ import absolute_import, division, print_function
from RandomEnsembleClustering import RandomClusteringClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

plt.style.use(["ggplot"])


def get_breast_cancer_data(test_size=0.5, random_state=None):
    """
    load data and split it into train and test set
    """
    x, y = load_breast_cancer(True)
    # reverse y (original y: 0: malignant, 1: benign.
    # but we want `malignant` to be the True class
    y = np.logical_not(y)
    train_x, test_x, train_y, test_y = (
        train_test_split(x, y, test_size=test_size, random_state=random_state))

    return train_x, train_y, test_x, test_y


def validate(data, n_probes=20, probes=None, n_estimators=50,
             probe_random_state=None, **kwargs):
    """

    Parameters
    ----------
    data: (train_x, train_y, test_x, test_y)
        the returned results from the get_breast_cancer_data() function.
        can also be data acquired else where passed in this order.
        If probes is provided, train_y won't be used (so it can be any
        arbitrary object)

    probes: array-like of shape = [n_probes, n_features], optional
        if probes are provided, use them to fit the model
        otherwise choose some from the data

    n_probes: integer, optional (default=20)
        number of probes to use to fit the model.
        if probes is not provided, randomly select n_probes sample in the
        true class for fitting the model.
        if probes is provided, this argument has no effect

    probe_random_state: integer, optional (default=None)
        The random state used for selecting probes. This probe_random_state is
        different from the random state for the RandomClusteringClassifier.
        The latter should be passed as a keyward argument `random_state`.
        This argument has no effect when probes is provided.

    **kwargs: keyword argument to be passed to the RandomClusteringClassifier.

    Return
    ------
    A dictionary of model, probes, and prediction results

    """

    train_x, train_y, test_x, test_y = data
    # if "probes" are not provided, get probes from train_x
    if probes is None:
        tr_true = train_x[train_y == 1]
        n_probes = min(n_probes, len(tr_true))
        probes = resample(tr_true, n_samples=n_probes, replace=False,
                          random_state=probe_random_state)

    n_probes = len(probes)
    # print out some conditions for the experiments
    n_unknown = len(train_x) - n_probes
    message = "Fitting {} models with {} 'True' samples and ".format(
              n_estimators, n_probes)
    message += "{} unknown samples \n".format(n_unknown)
    print(message)

    # initialize the RandomClusteringClassifier
    rcc = RandomClusteringClassifier(n_estimators=n_estimators, **kwargs)

    # fit the classifier using only train_x and probes (train_y is NOT used)
    # here train_x is the unlabelled data, probes are some known data in the
    # True class
    rcc.fit(train_x, probes)

    # predict on test_x, the test set that hasn't been seen by the rcc model
    pred = rcc.predict(test_x)
    proba = rcc.predict_proba(test_x)  # for other metrics

    print("Validating on {} samples in the test set".format(len(test_x)))
    print_results(test_y, pred)  # print out some test results

    return {"y_true": test_y,
            "y_pred": pred,
            "y_proba": proba,
            "probes": probes,
            "model": rcc}


def print_results(y_true, y_pred):
    """ Print out and plot some test results

    Parameters
    ----------
    y_true: array-like of shape = [n_test_samples]
        A boolean array of the actual classes (1: True)

    y_true: array-like of shape = [n_test_samples]
        A boolean array of the predicted classes
    """
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Accuracy: {:.2f} %".format(100 * accuracy_score(y_true, y_pred)))

    return


def plot_confusion_matrix(y_true, y_pred, labels=None,
                          add_title=True, norm=False):
    """
    Parameters
    ----------
    y_true: array-like of shape = [n_test_samples]
        A boolean array of the actual classes (1: True)

    y_true: array-like of shape = [n_test_samples]
        A boolean array of the predicted classes

    labels: label name of each class, optional (default None)
        if None, use [0, 1] as labels

    add_title: boolean, optional (default True)
        whether to add titles in the figure. title would be
        the accuracy, and the precision and recall of the `True` class

    norm: boolean, optional (default False)
        If True, plot normalized confusion matrix in which each cell
        will be normalized to the total counts in the actual class

    """
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or range(len(cm))
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    if norm:
        cm = cm / np.sum(cm, axis=1)[:, np.newaxis]
    # cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
    cmap = "Blues"
    plt.figure(dpi=80)

    fmt = "g" if not norm else ".3f"
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=fmt, annot_kws={"size": 14}, cmap=cmap)
    plt.xlabel("Predicted", size=16)
    plt.ylabel("Actual", size=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yticks(rotation="horizontal")
    if add_title:
        plt.title("Accuracy: {:.2f}\nprecision: {:.2f}, recall: {:.2f}".format(
                  accuracy, precision, recall))
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    data = get_breast_cancer_data(test_size=0.4, random_state=5)
    res = validate(data, n_probes=20, n_estimators=200, random_state=5,
                   verbose=1)
    plot_confusion_matrix(res["y_true"], res["y_pred"],
                          labels=["Benign", "Malignant"])
    plt.show()
