import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, length, channel):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(length, channel)),
            # layers.Reshape((length, 1, channel)),
            # layers.Dense(96),
            layers.Reshape((length, channel, 1)),
            layers.Conv2D(32, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 3)),
            layers.Conv2D(64, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 3)),
            layers.Conv2D(64, (1, 1), activation='sigmoid', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 5)),
        ])
        self.decoder = tf.keras.Sequential([
            layers.UpSampling2D((2, 5)),
            layers.Conv2D(64, (1, 1), activation='relu', padding='same', strides=1),
            layers.UpSampling2D((2, 3)),
            layers.Conv2D(64, (1, 1), activation='relu', padding='same', strides=1),
            layers.UpSampling2D((2, 3)),
            layers.Conv2D(32, (1, 1), activation='relu', padding='same', strides=1),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (1, 1), activation='linear', padding='same', strides=1),
            # layers.Reshape((length, 1, 96)),
            # layers.Dense(channel),
            layers.Reshape((length, channel))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
