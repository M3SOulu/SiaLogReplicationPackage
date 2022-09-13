from typing import Dict
import numpy as np

from base_classes import BaseDistributionMonitor


class GaussianDistributionMonitor(BaseDistributionMonitor):

    def __init__(self, embedding_model, components=15):
        from sklearn.mixture import GaussianMixture as GMM
        super().__init__(embedding_model)
        self.gmm = GMM(components)
        self.normal_distributions = None

    def fit(self, data_table: Dict[int, np.ndarray]):
        self.gmm.fit(np.concatenate([self.embed(data_table[0]), self.embed(data_table[1])], axis=0))

    def fitness_score(self, data_table: Dict[int, np.ndarray]):
        return self.gmm.score_samples(np.concatenate([self.embed(data_table[0]), self.embed(data_table[1])], axis=0))
