# -*- coding: utf-8 -*-
# @Author: C. Marcus Chuang

from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KmeansLabeler(object):
    """
    Use some probes (samples with known label) and k-means clustering to label
    unknown samples that are similar to the probes.
    """

    def __init__(self, k=2, n_jobs=-2, **kwargs):
        """ Initialization

        Parameters
        ----------
        k : integer, optional (default=2)
            number of clusters in k-means model

        n_jobs : integer, optional (default=1)
            The number of jobs to run in parallel for both `fit` and `predict`.
            If -1, then the number of jobs is set to the number of cores.

        **kwargs:
            keyword arguments to be passed to `KMeans` model
        """
        self.model = KMeans(n_clusters=k, n_jobs=n_jobs, **kwargs)
        self._is_fitted = False

    def fit(self, X, probes):
        """ Fit the model usng X and probes

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            the training input samples (label unknown)

        probes: array-like of shape = [n_probes, n_features]
            the taining input probes whose labels are `True`

        Returns
        -------
        None
        """
        self.model.fit(X)
        probe_pred = self.model.predict(probes)
        class_count = Counter(probe_pred)
        self.class_proba = {k: v/len(probes) for k, v in class_count.items()}
        self.target_label = 1 * max(class_count, key=lambda x: class_count[x])
        self._is_fitted = True
        # print(self.class_proba)

        return

    def predict_proba(self, X):
        """ Predict the probablility of each sample in X is of the same label
            of probes.

        Probability is calculated based on the percentage of probes in a
        given cluster

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            the samples to be labeled

        Returns
        -------
        proba : an 1-D array

        """
        if not self._is_fitted:
            raise NotFittedError("This model has not been fitted yet")
        pred = self.model.predict(X)
        proba = np.array([self.class_proba.get(x, 0) for x in pred])
        # proba = np.array(map(lambda x: self.class_proba[x], pred))

        return proba

    def predict(self, X):
        """ Predict whether each sample in X is of the same label of probes

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            the samples to be labeled

        Returns
        -------
        pred : an 1-D boolean array

        """
        if not self._is_fitted:
            raise NotFittedError("This model has not been fitted yet")
        # proba = self.predict_proba(X)
        pred = (self.model.predict(X) == self.target_label)

        return pred

    def fit_and_predict(self, X, probes):
        """ Fit and predict the label of X using X and probes

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            the training input samples (label unknown)

        probes: array-like of shape = [n_probes, n_features]
            the taining input probes whose labels are `True`

        Returns
        -------
        pred : an 1-D boolean array
        """
        self.fit(X, probes)

        return self.predict(X)


class RandomClusteringClassifier(object):
    """
    Use some probes (samples with known label) and ensemble of k-means models
    to label (classifiy) unknown samples that are similar to the probes.

    Labeling are based on majority votes from all the models.
    For each model, both probes and the unknown samples are resampled by
    bootstrap methods. In addition, a subset of the features are then randomly
    selected for k-mean clustering.

    """
    def __init__(self, k=2, n_estimators=50, max_features=5, min_features=2,
                 scale_features=True, verbose=0, random_state=None,
                 voting_rule="hard", **kwargs):
        """
        Parameters
        ----------
        k : integer, optional (default=2)
            number of clusters in k-means model

        n_estimators: integer, optional (default=50)
            number of k-means model to run

        max_features: integer, optional (default=5)
            maximum number of features to use in each model; would be
            overwritten if max_features is greater than number of features in
            the data.

        min_features: integer, optional (default=2)
            minimum number of features to use in each model

        scale_features: boolean, optional (default=True)
            whether to scale each features. If true, a StandardScaler will be
            used to scale the each feature based on the data in `X`. The same
            scaling will be applied to `probes` and the test data

        voting_rule: string, optional (default=hard)
            available: `hard` and `soft`
            The voting rules for ensemble prediction.
            Hard: prediction from each model is either 0 (False) or 1 (True)
            Soft: prediction from each model is the probability (between 0 & 1)
            The final prediction is the sum of the prediction from all models


        # to do: assert max > min

        """

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_features = min_features
        self.scale_features = scale_features
        self.verbose = verbose
        self._is_fitted = False
        self.random_state = random_state
        self.k = k
        if voting_rule not in ("soft", "hard"):
            raise ValueError
        self.voting_rule = voting_rule  # hard: 0-1, soft: proba from each md
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
                if i == 0 or (i+1) % (self.n_estimators//10) == 0:
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
        n_fea = self.np_random.randint(self.min_features, max_features+1)
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
            if self.voting_rule == "hard":
                pred = model.predict(self._slicing_col(X_copy, feat_ind))
            else:
                pred = model.predict_proba(self._slicing_col(X_copy, feat_ind))

            predicted.append(pred)

        return np.array(predicted)


class NotFittedError(Exception):
    pass
