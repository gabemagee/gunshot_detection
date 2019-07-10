from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.datasets import load_iris

tf.enable_eager_execution()


data = load_iris()
X = data['data']
y = data['target']


def make_dataset(X_data, y_data, n_splits):
    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train, y_train, X_test, y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64, tf.float64, tf.float64, tf.float64))


dataset = make_dataset(X, y, 3)
print("hi")
i = 0
for X_train, y_train, X_test, y_test in tfe.Iterator(dataset):
    print(i)
    i = i + 1
    # print(X_train,y_train,X_test,y_test)
