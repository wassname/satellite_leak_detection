
import numpy as np


def calculate_result_class(y_pred, y_true, thresh=0.5, words=False):
    """Calculate int results classes. Here 'false positive' = 'fp' = bin('01') = 1 etc."""
    dic = {0: 'fn', 1: 'fp', 2: 'tn', 3: 'tp'}
    y_pred = np.array(y_pred) > thresh
    y_true = np.array(y_true) > thresh

    result_class_bools = np.array(list(zip(y_pred == y_true, y_pred))) * 1
    # convert to binary
    result_class_binary = ['0b' + ''.join(d) for d in result_class_bools.astype(str)]
    result_class = [int(d, 2) for d in result_class_binary]

    if words:
        result_class = [dic[d] for d in result_class]
    return result_class


assert calculate_result_class([0, 1, 0, 1], [1, 0, 0, 1], words=False) == [0, 1, 2, 3]
assert calculate_result_class([0, 1, 0, 1], [1, 0, 0, 1], words=True) == ['fn', 'fp', 'tn', 'tp']
assert calculate_result_class([False, True, False, True], [True, False, False, True], words=False) == [0, 1, 2, 3]


from io import StringIO
import pandas as pd
import numpy as np
from sklearn import metrics


def parse_classification_report(classification_report):
    """Parse a sklearn classification report to a dict."""
    return pd.read_fwf(
        StringIO(classification_report),
        index_col=0,
        colspecs=[(0, 12), (12, 22), (22, 32), (32, 42), (42, 52)]
    ).dropna()

# test
s = metrics.classification_report(np.random.random(100) > 0.5, np.random.random(100) > 0.5)
d = parse_classification_report(s).to_dict()
assert isinstance(d, dict)


import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
import collections


def find_best_dummy_classification(X, y, test_size=0.3, random_state=0, thresh=0.5, target_names=None, n=1):
    """Try all dummy models."""
    X = X.reshape((len(X) ,-1))
    # y = y.reshape((len(y) ,-1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    dummy_scores = []
    for i in range(n):
        for strategy in ['most_frequent', 'uniform', 'prior', 'stratified']:
            clf = DummyClassifier(strategy=strategy)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = clf.score(X_test, y_test)

            matthews_corrcoef=sklearn.metrics.matthews_corrcoef(y_test > thresh, y_pred > thresh)

            report=parse_classification_report(sklearn.metrics.classification_report(y_test > thresh, y_pred > thresh, target_names=target_names))

            dummy_scores.append(
                collections.OrderedDict(
                    strategy='classifier_' + strategy,
                    matthews_corrcoef=matthews_corrcoef,
                    score=score,
                    report=report
                )
            )

        for strategy in ['mean', 'median']:
            clf=DummyRegressor(strategy=strategy)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            score=clf.score(X_test, y_test)

            matthews_corrcoef=sklearn.metrics.matthews_corrcoef(y_test > thresh, y_pred > thresh)

            report=parse_classification_report(sklearn.metrics.classification_report(y_test > thresh, y_pred > thresh, target_names=target_names))

            dummy_scores.append(
                collections.OrderedDict(
                    strategy='regressor_' + strategy,
                    matthews_corrcoef=matthews_corrcoef,
                    score=score,
                    report=report
                )
                )

    df=pd.DataFrame(dummy_scores)
    df=df.sort_values('matthews_corrcoef', ascending=False)
    return df, df[:1].iloc[0].to_dict()

# test
# import numpy as np
# X=np.random.random((100,10,10,10))
# y=np.random.random((100))>0.5
# df, dummy = find_best_dummy_classification(X,y,n=100)
# df_dummies.groupby('strategy').mean()
