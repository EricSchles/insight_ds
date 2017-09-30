# -*- coding: utf-8 -*-
# @Author: C. Marcus Chuang
# @Date:   2017-09-27 19:29:38
# @Last Modified by:   C. Marcus Chuang
# @Last Modified time: 2017-09-30 01:48:49

from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import fbeta_score
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import recall_score, accuracy_score, precision_score


class NotFittedError(Exception):
    pass


class KmeansLabeler(object):

    def __init__(self, k=2, n_jobs=3, **kwargs):
        self.model = KMeans(n_clusters=k, n_jobs=n_jobs, **kwargs)
        self._is_fitted = False

    def fit(self, X, probes):
        self.model.fit(X)
        probe_pred = self.model.predict(probes)
        class_count = Counter(probe_pred)
        self.true_label = 1 * max(class_count, key=lambda x: class_count[x])
        self._is_fitted = True

        return

    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError("This model has not been fitted yet")
        pred = self.model.predict(X)

        return pred == self.true_label


class EnsembleKmeansClassifier(object):

    def __init__(self, k=2, n_estimators=50, max_features=5, min_features=2,
                 scale_features=True, verbose=0, random_state=None, **kwargs):

        # to do: assert max > min
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_features = min_features
        self.scale_features = scale_features
        self.verbose = verbose
        self._is_fitted = False
        self.random_state = random_state
        self.k = k
        if random_state:
            self.np_random = np.random.RandomState(random_state)
        else:
            self.np_random = np.random.RandomState()
        kwargs["random_state"] = random_state
        self._kmean_kwargs = kwargs

    def fit(self, X, probes, X_test=None, Y_test=None):
        # to do: add an option to predict and output results for each model
        #        or predict on oob samples
        self.models = [KmeansLabeler(k=self.k, **self._kmean_kwargs)
                       for i in range(self.n_estimators)]
        self.feature_ind = []
        X_copy, probes_copy = self._process(X, probes)

        for i, model in enumerate(self.models):
            if self.verbose:
                print("Model #  {}  / {}".format(i+1, self.n_estimators))
            feat_ind = self._select_features(n_cols=X_copy.shape[1])
            # print(feat_ind)
            self.feature_ind.append(feat_ind)
            X_resample = self._bootstrap(X_copy)
            probes_resample = self._bootstrap(probes_copy)
            X_subset = self._slicing_col(X_resample, feat_ind)
            probes_subset = self._slicing_col(probes_resample, feat_ind)
            model.fit(X_subset, probes_subset)
            # if y: print res

        self._is_fitted = True

        return

    def predict_proba(self, X, test_set=None):
        y_pred = self._predict(X, test_set=None)
        y_ensemble = y_pred.sum(axis=0)
        y_proba = y_ensemble / len(y_pred)

        return y_proba

    def predict(self, X, test_set=None, threshold=0.5):
        y_proba = self.predict_proba(X, test_set=test_set)
        return y_proba >= threshold

    def _bootstrap(self, *arr):
        return resample(*arr, replace=True, random_state=self.np_random)

    def _slicing_col(self, X, col_ind):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, col_ind]
        return X[:, col_ind]

    def _select_features(self, n_cols):
        """
        randomly select features

        Returns:
            a sorted numpy array of selected column indices
        """
        max_features = min(n_cols, self.max_features)
        n_fea = self.np_random.randint(self.min_features, max_features)
        selected_col_ind = self.np_random.choice(range(n_cols),
                                                 size=n_fea, replace=False)
        selected_col_ind.sort()

        return selected_col_ind

    def _process(self, X, probes):
        if self.scale_features:
            self.scaler = StandardScaler()
            X_copy = self.scaler.fit_transform(X)
            probes_copy = self.scaler.transform(probes)
            return X_copy, probes_copy

        return X.copy(), probes.copy()

    def _predict(self, X, test_set=None):
        """
        """
        if not self._is_fitted:
            raise NotFittedError("This model has not been fitted yet")
        if self.scale_features:
            X_copy = self.scaler.transform(X)
        else:
            X_copy = X.copy()
        predicted = []

        for model, feat_ind in zip(self.models, self.feature_ind):
            pred = model.predict(self._slicing_col(X_copy, feat_ind))
            # pred = (pred == model.true_label)
            predicted.append(pred)

        return np.array(predicted)

