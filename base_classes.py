import csv
from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score


class BaseModel:

    def __init__(self, embedding_model: tf.keras.Model):
        self.__embedding_model = embedding_model

    def embed(self, x):
        return self.__embedding_model.predict(tf.keras.preprocessing.sequence.pad_sequences(x))

    @property
    def embedding_model(self):
        return self.__embedding_model


class BaseVisualizer(BaseModel):
    pass


class TensorflowProjectorVisualizer(BaseVisualizer, ABC):

    def export(self, output_vectors_path=None, output_metadata_path=None):
        if not output_vectors_path:
            output_vectors_path = f"out/{type(self).__name__}_projector_vectors.tsv"
        if not output_metadata_path:
            output_metadata_path = f"out/{type(self).__name__}_projector_metadata.tsv"
        with open(output_vectors_path, "w") as vectors, open(output_metadata_path, "w") as metadata:
            v_tsv = csv.writer(vectors, delimiter="\t")
            m_tsv = csv.writer(metadata, delimiter="\t")
            self.write_tsv(v_tsv, m_tsv)

        print(f"{type(self).__name__}: metadata exported to file {output_metadata_path}")
        print(f"{type(self).__name__}: vectors exported to file {output_vectors_path}")

    @abstractmethod
    def write_tsv(self, v_tsv, m_tsv):
        pass


class BaseClassifier(BaseModel, ABC):

    def __init__(self, embedding_model, prefer_balanced_data=False, gmm_balancing_components=1):
        super().__init__(embedding_model)
        self.prefer_balanced_data = prefer_balanced_data
        self.gmm_balancing_components = gmm_balancing_components

    def evaluate(self, data_table) -> Dict[str, float]:
        y_true = np.concatenate([np.zeros(len(data_table[0])), np.ones(len(data_table[1]))])
        y_pred = np.concatenate([self.predict(data_table[0]), self.predict(data_table[1])])
        return {
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        }

    def prepare_data_table(self, data_table: Dict[int, np.ndarray]):
        if self.prefer_balanced_data:
            positive_samples = self.embed(data_table[1])
            balancing_positive_samples = generate_sample(positive_samples, len(data_table[0]) - len(data_table[1]), self.gmm_balancing_components)
            x = np.concatenate([self.embed(data_table[0]), positive_samples, balancing_positive_samples],
                               axis=0)
            y = np.concatenate(
                [np.zeros(len(data_table[0])), np.ones(len(data_table[1]) + len(balancing_positive_samples))])
        else:
            x = np.concatenate([self.embed(data_table[0]), self.embed(data_table[1])], axis=0)
            y = np.concatenate([np.zeros(len(data_table[0])), np.ones(len(data_table[1]))])
        return x, y

    @abstractmethod
    def fit(self, data_table: Dict[int, np.ndarray]):
        pass

    @abstractmethod
    def predict(self, x: Iterable) -> np.ndarray:
        pass


class BaseDistributionMonitor(BaseModel, ABC):
    @abstractmethod
    def fit(self, data_table: Dict[int, np.ndarray]): pass

    @abstractmethod
    def fitness_score(self, data_table: Dict[int, np.ndarray]): pass


def generate_sample(distribution_samples, size, gmm_components):
    from sklearn.mixture import GaussianMixture as GMM
    gmm = GMM(gmm_components)
    gmm.fit(distribution_samples)
    return gmm.sample(size)[0]
