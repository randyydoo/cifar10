import cv2 as cv
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import matplotlib.pyplot as plt
import numpy as np

cifar = tf.keras.datasets.cifar10
# 32x32 images
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# rescale pixel values
x_train , y_train = x_train / 255.0, y_train / 255.0
inputs = KL.input(shape = (32,32,3))

