import tensorflow as tf
import keras_tuner as kt


# distribute_strategy = tf.distribute.MirroredStrategy()


def SiameseNetwork(embedding_model):
    # with distribute_strategy.scope():
    x1 = tf.keras.Input(shape=(None,))
    x2 = tf.keras.Input(shape=(None,))
    y1 = embedding_model(x1)
    y2 = embedding_model(x2)
    dot = tf.keras.layers.Dot(axes=-1)([y1, y2])
    y = tf.keras.layers.Activation("sigmoid")(dot)
    # y = tf.keras.layers.Mean()(y)
    # y = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2]))(y)
    # y = tf.keras.layers.Reshape((1,))(y)
    model = tf.keras.Model(inputs=[x1, x2], outputs=[y])
    model.compile("adam", loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def embedding_hdfs(embedding_input_dim):
    x = tf.keras.Input(shape=(None,))
    y = tf.keras.layers.Embedding(embedding_input_dim, 128)(x)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(y)
    y = tf.keras.layers.Dense(256, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128)(y)
    model = tf.keras.Model(inputs=[x], outputs=[y])
    return model


def embedding_bgl(embedding_input_dim):
    x = tf.keras.Input(shape=(None,))
    y = tf.keras.layers.Embedding(embedding_input_dim, 128)(x)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(y)
    y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128)(y)
    model = tf.keras.Model(inputs=[x], outputs=[y])
    return model


def embedding_hadoop(embedding_input_dim):
    x = tf.keras.Input(shape=(None,))
    y = tf.keras.layers.Embedding(embedding_input_dim, 64)(x)
    # y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(y)
    # y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(y)
    # y = tf.keras.layers.GRU(64, return_sequences=True)(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.GRU(64)(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(64, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(64, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(64, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dropout(0.5)(y)
    # y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dense(128)(y)
    model = tf.keras.Model(inputs=[x], outputs=[y])
    return model


def Embedding(embedding_input_dim, model_size=None):
    x = tf.keras.Input(shape=(None,))
    # y = tf.keras.layers.Embedding(embedding_input_dim, 64)(x)
    # # y = tf.keras.layers.LSTM(192, return_sequences=True)(y)
    # # y = tf.keras.layers.LSTM(64, return_sequences=True)(y)
    # y = tf.keras.layers.LSTM(64)(y)
    # # a = tf.keras.layers.Conv1D(1, 1)(y)
    # # a = tf.keras.layers.Softmax(-2)(a)
    # # y = tf.keras.layers.Multiply()([y, a])
    # # y = tf.keras.layers.GlobalAvgPool1D()(y)
    # y = tf.keras.layers.Dense(256, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    # y = tf.keras.layers.Dense(128)(y)
    y = tf.keras.layers.Embedding(embedding_input_dim, 128)(x)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(y)
    y = tf.keras.layers.Dense(256, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128, activation=tf.nn.silu)(y)
    y = tf.keras.layers.Dense(128)(y)
    model = tf.keras.Model(inputs=[x], outputs=[y])
    return model

    # with distribute_strategy.scope():

    # return tf.keras.models.Sequential([
    #     tf.keras.layers.Embedding(30, 24, mask_zero=True),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #     # tf.keras.layers.LSTM(128, return_sequences=True),
    #     # tf.keras.layers.LSTM(64, return_sequences=True),
    #     # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    #     tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
    #     tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
    #     tf.keras.layers.Dense(64),
    # ])


class SiameseHyperModel(kt.HyperModel):

    def __init__(self, embedding_input_dim):
        self.embedding_input_dim = embedding_input_dim
        super().__init__()

    def build(self, hp):
        emb_model = self.embedding_model(hp)
        x1 = tf.keras.Input(shape=(None,))
        x2 = tf.keras.Input(shape=(None,))
        y1 = emb_model(x1)
        y2 = emb_model(x2)
        dot = tf.keras.layers.Dot(axes=-1)([y1, y2])
        y = tf.keras.layers.Activation("sigmoid")(dot)
        model = tf.keras.Model(inputs=[x1, x2], outputs=[y])
        model.compile("adam", loss=tf.keras.losses.binary_crossentropy)
        return model

    def embedding_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(self.embedding_input_dim, hp.Int("embedding dim", 16, 64, 8)))
        for i in range(hp.Int("S2S LSTMs", 0, 3)):
            model.add(tf.keras.layers.LSTM(hp.Int(f"LSTM {i} units", 16, 64, 8), return_sequences=True))
        model.add(tf.keras.layers.LSTM(hp.Int(f"final LSTM units", 32, 128, 8)))
        for i in range(hp.Int("S2S FCs", 1, 5)):
            model.add(tf.keras.layers.Dense(hp.Int(f"FC {i} units", 32, 128, 8)))
        model.add(tf.keras.layers.Dense(hp.Int("output dimensions", 16, 128, 16)))
        return model


def EmbeddingAndClassifier(embedding_input_dim):
    x = tf.keras.Input(shape=(None,))
    l = tf.keras.layers.Embedding(embedding_input_dim, 128)(x)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(l)
    l = tf.keras.layers.Dropout(0.5)(l)
    l = tf.keras.layers.Dense(128, activation=tf.nn.silu)(l)
    l = tf.keras.layers.Dropout(0.5)(l)
    l = tf.keras.layers.Dense(128, activation=tf.nn.silu)(l)
    l = tf.keras.layers.Dropout(0.5)(l)
    l = tf.keras.layers.Dense(128, activation=tf.nn.silu)(l)
    l = tf.keras.layers.Dropout(0.5)(l)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(l)
    model = tf.keras.Model(inputs=[x], outputs=[l, y])
    return model, tf.keras.Model(inputs=[x], outputs=[y])


def LatentSiamesation(model):
    from metrics import F1Score
    x1 = tf.keras.Input(shape=(None,))
    x2 = tf.keras.Input(shape=(None,))
    l1, y1 = model(x1)
    l2, _ = model(x2)
    ldot = tf.keras.layers.Dot(axes=-1)([l1, l2])
    ys = tf.keras.layers.Activation("sigmoid", name="latent")(ldot)
    model = tf.keras.Model(inputs=[x1, x2], outputs=[ys, y1])
    model.compile("adam", loss=tf.keras.losses.binary_crossentropy,
                  metrics={"model": F1Score()})
    return model
