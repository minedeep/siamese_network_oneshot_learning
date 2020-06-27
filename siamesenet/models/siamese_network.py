import os
import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


tf.keras.backend.clear_session()  # For easy reset of notebook state.

class SiameseNetwork(tf.keras.Model):
    """
    Construct Siamese network
    """

    def __init__(self, img_size, way=2):
        """
            img_size: tuple (w, h, c)
            self_way: number of classes used in training
        """
        super(SiameseNetwork, self).__init__()
        self.width, self.height, self.channel = img_size
        self.way = way

        # Conv_net 
        self.encoder = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=64, kernel_size=10, padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D((2,2)),

                tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D((2,2)),

                tf.keras.layers.Conv2D(filters=128, kernel_size=4, padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D((2,2)),


                tf.keras.layers.Conv2D(filters=128, kernel_size=4, padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                
                tf.keras.layers.Flatten() ])
        
        # Prediction layer
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])

    @tf.function
    def call(self, support, query, labels):
        # Number of examples in each batch
        n = support.shape[0]

        # Concatenate and feed to the encoder
        cat = tf.concat([support, query], axis=0)
        encoded_cat = self.encoder(cat)

        #Split into two encoded images
        en_support = encoded_cat[:n]
        en_query = encoded_cat[n:]

        # L1-distance
        l1_dist = tf.abs(en_support - en_query)

        # Prediction layer
        prediction = self.dense(l1_dist)

        # loss and accuracy
        loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_true=labels, y_pred=prediction))
        labels_pred = tf.cast(tf.argmax(tf.reshape(prediction, [-1, self.way]), -1), tf.float32)
        labels_true = tf.tile(tf.range(0, self.way, dtype=tf.float32), [n//self.way**2])
        eq = tf.cast(tf.equal(labels_pred, labels_true), tf.float32)
        acc = tf.reduce_mean(eq)

        return loss, acc

    def save(self, save_dir):

        """
        input: save_dir: path to where to store the trained model
        return: None
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        encoder_path = os.path.join(save_dir, 'encoder.h5')
        dense_path = os.path.join(save_dir, 'dense.h5')
        self.dense.save(dense_path)
        self.encoder.save (dense_path)

    def load(self, dir):

        """
        dir: path where the trained model stored (.h5 file)
        return None
        """

        encoder_path = os.path.join(dir, 'encoder.h5')
        self.encoder(tf.zeros([1, self.width, self.height, self.channel]))
        self.encoder.load_weights(encoder_path)
        dense_path = os.path.join(dir, 'dense.h5')
        self.dense(tf.zeros([1, 4608]))
        self.dense.load_weights(dense_path)

