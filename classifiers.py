from typing import Dict, Iterable

import numpy as np

from base_classes import BaseClassifier


class KNearestNeighbors(BaseClassifier):

    def __init__(self, embedding_model, n_neighbors=10):
        from sklearn.neighbors import KNeighborsClassifier as KNN
        super().__init__(embedding_model, False)
        self.knn = KNN(n_neighbors, n_jobs=-1)

    def fit(self, data_table: Dict[int, np.ndarray]):
        self.knn.fit(*self.prepare_data_table(data_table))

    def predict(self, x: Iterable) -> np.ndarray:
        return self.knn.predict(self.embed(x))


class SupportVectorMachine(BaseClassifier):

    def __init__(self, embedding_model):
        from sklearn.svm import SVC as SVM
        super().__init__(embedding_model, False)
        self.svm = SVM()

    def fit(self, data_table: Dict[int, np.ndarray]):
        self.svm.fit(*self.prepare_data_table(data_table))

    def predict(self, x: Iterable) -> np.ndarray:
        return self.svm.predict(self.embed(x))


class NeuralNetwork(BaseClassifier):

    def __init__(self, embedding_model, epochs=128, batch_size=1024):
        import tensorflow as tf
        super().__init__(embedding_model, False)
        self.epochs = epochs
        self.batch_size = batch_size
        self.mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2()),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2()),
        ])
        self.mlp.compile("sgd", loss=tf.keras.losses.binary_crossentropy)

    def fit(self, data_table: Dict[int, np.ndarray]):
        self.mlp.fit(*self.prepare_data_table(data_table), batch_size=self.batch_size, epochs=self.epochs, verbose=0)

    def predict(self, x: Iterable) -> np.ndarray:
        return np.round(self.mlp.predict(self.embed(x)))


class LogisticRegression(BaseClassifier):

    def __init__(self, embedding_model):
        from sklearn.linear_model import LogisticRegression as LR
        super().__init__(embedding_model, False)
        self.lr = LR(n_jobs=-1, max_iter=255)

    def fit(self, data_table: Dict[int, np.ndarray]):
        self.lr.fit(*self.prepare_data_table(data_table))

    def predict(self, x: Iterable) -> np.ndarray:
        return self.lr.predict(self.embed(x))
