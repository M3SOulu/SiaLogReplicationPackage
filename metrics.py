import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, class_id=None, **kwargs):
        super().__init__(name="f1_score", **kwargs)
        self.precision = tf.keras.metrics.Precision(class_id=class_id)
        self.recall = tf.keras.metrics.Recall(class_id=class_id)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class TopK(tf.keras.metrics.Metric):

    def __init__(self, k=3, **kwargs):
        super().__init__(name=f"top_{k}", **kwargs)
        self.topk = tf.keras.metrics.SparseTopKCategoricalAccuracy(k)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.boolean_mask(y_pred, y_true > 0)
        y_true = tf.boolean_mask(y_true, y_true > 0)
        self.topk.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.topk.result()

    def reset_state(self):
        self.topk.reset_state()
