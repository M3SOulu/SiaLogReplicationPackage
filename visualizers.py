from base_classes import TensorflowProjectorVisualizer


class EventEmbeddingProjectorVisualizer(TensorflowProjectorVisualizer):
    def __init__(self, embedding_model, dataset):
        super().__init__(embedding_model)
        self.dataset = dataset

    def event_embedding_layer_weights(self):
        from tensorflow.keras.layers import Embedding
        for layer in self.embedding_model.layers:
            if type(layer) == Embedding:
                return layer.weights[0].numpy()

    def event_type(self, event_id):
        if event_id == 0:
            return "padding"
        elif event_id in self.dataset.negative_events:
            return "normal_only"
        else:
            return "normal_anomaly"

    def write_tsv(self, v_tsv, m_tsv):
        m_tsv.writerow(["event_id", "event_type"])
        for e, v in enumerate(self.event_embedding_layer_weights()):
            m_tsv.writerow([str(e), self.event_type(e)])
            v_tsv.writerow(v)


class SequenceEmbeddingProjectorVisualizer(TensorflowProjectorVisualizer):
    def __init__(self, embedding_model, dataset):
        super().__init__(embedding_model)
        self.dataset = dataset

    def tsv_iter(self, x):
        return zip(x, self.embed(x))

    def write_tsv(self, v_tsv, m_tsv):
        m_tsv.writerow(["sequence", "set", "label", "set_label"])

        for seq, vec in self.tsv_iter(self.dataset.train_data_table[0]):
            v_tsv.writerow(vec)
            m_tsv.writerow([str(seq), "train", "non-anomaly", "train_non"])

        for seq, vec in self.tsv_iter(self.dataset.train_data_table[1]):
            v_tsv.writerow(vec)
            m_tsv.writerow([str(seq), "train", "anomaly", "train_ano"])

        for seq, vec in self.tsv_iter(self.dataset.test_data_table[0]):
            v_tsv.writerow(vec)
            m_tsv.writerow([str(seq), "test", "non-anomaly", "test_non"])

        for seq, vec in self.tsv_iter(self.dataset.test_data_table[1]):
            v_tsv.writerow(vec)
            m_tsv.writerow([str(seq), "test", "anomaly", "test_ano"])
