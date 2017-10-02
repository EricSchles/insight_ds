# Random Clustering Classifier
This is a classifier I wrote in Python for [my project at Insight Data Science](https://medium.com/@marcus.chchuang/insight-project-ebe38a8bf9ee). 

It is a very interesting and challenging project for a non-profit organization. The goal is to identify some “high value targets” from the samples. However, we only have a small amount of samples that are in the `True` class (referred to as **probes**), and we have no information about other samples (they could either be `True` or `False`), nor do we know the distribution of these two classes in the samples.

The classifier is based on k-means clustering with `k=2` clusters:

```
class KmeansLabeler(object):
    """
    Use some probes (samples with known label) and k-means clustering to label
    unknown samples that are similar to the probes.
    """
``` 

Use k-means clustering to cluster the data into 2 different clusters and then use the number of **probes** in each cluster to determine which cluster is `True`. In 2-D space, it looks like this:
![alt text](https://cdn-images-1.medium.com/max/800/1*j5B0e_QLT-n6r6oAuzoG3w.png)

I further employed **Bagging** to reduce the variance of the prediction:

```
class RandomClusteringClassifier(object):
    """
    Use some probes (samples with known label) and ensemble of k-means models
    to label (classifiy) unknown samples that are similar to the probes.

    Labeling are based on the majority vote from all the models.
    For each model, both the probes and the unknown samples are resampled by
    bootstrap methods. In addition, for each k-means clustering model, the 
    features used are a randomly selected subset from all features.
    """

    def __init__(self, k=2, n_estimators=50, max_features=5, min_features=2,
                 scale_features=True, verbose=0, random_state=None,
                 voting_rule="hard", **kwargs):
    ...
```
![alt text](https://cdn-images-1.medium.com/max/800/1*OOTcakHnjIElhLmKbRSWpw.png)


The classifiers use similar syntax as `sklearn`'s classifiers with methods such as `fit`, `predict`, and `predict_proba`.


For more information about the classifier, please see the docstring.

For more information about the project, see the [blog post](https://medium.com/@marcus.chchuang/insight-project-ebe38a8bf9ee)

