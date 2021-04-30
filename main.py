import os 
import cv2
import pickle
import numpy as np
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, LeakyReLU, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Embedding, Reshape, Activation
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, multiply, UpSampling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

import tensorflow as tf
print(tf.__version__)


def discriminator(input_shape=(112, 112, 3)):
    #weight initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)
    
    # convolutional block
    def conv_block(input_layer, filter_size, stride):
        x = Conv2D(filter_size, kernel_size = (3,3), padding='same', strides=stride, kernel_initializer = init)(input_layer)
        x = BatchNormalization(momentum = 0.9)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dropout(0.5)(x)
        return x
    
    # input image
    input_img = Input(shape=input_shape)
    
    x = Conv2D(32, kernel_size = (3, 3), strides = (1, 1), padding='same', kernel_initializer = init)(input_img)
    x = BatchNormalization(momentum = 0.9)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.5)(x)

    # downsample to 56 x 56 x 64
    x = conv_block(x, 64, (2, 2))
    # downsample to 28 x 28 x 128
    x = conv_block(x, 128, (2, 2))
#     x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
#     x = BatchNormalization(momentum = 0.9)(x)
#     x = LeakyReLU(alpha = 0.2)(x)
#     x = Dropout(0.5)(x)
    # downsample to 14 x 14 x 256
    x = conv_block(x, 256, (2, 2))
    # downsample to 7 x 7 x 512
    x = conv_block(x, 512, (2, 2))
#     x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
#     x = BatchNormalization(momentum = 0.9)(x)
#     x = LeakyReLU(alpha = 0.2)(x)
#     x = Dropout(0.5)(x)
    
    # flatten layer
    features = Flatten()(x)

    # binary classifier, image fake or real
    fake = Dense(1, activation='sigmoid', name='generation')(features)

    # multi-class classifier, image digit class
    aux = Dense(2, activation='softmax', name='auxiliary')(features)

    model = Model(inputs = input_img,  outputs = [fake, aux])

    return model
  
  
d = discriminator(input_shape=(112, 112, 3))
d.summary()

models = tf.keras.models
def generator(latent_dim = 100, n_classes = 2):
    init = RandomNormal(mean = 0.0, stddev = 0.02)
  
    # Input 1: class label input
    label_input = Input(shape = (1,))
    #print(label_input.shape)
    y = Embedding(n_classes, 100)(label_input)
    #print('Embedding Layer: ', y.shape)
    n_nodes = 7 * 7
    y = Dense(n_nodes, kernel_initializer = init)(y)
    #print('Dense 1: ', y.shape)
    y = Reshape((7, 7 ,1))(y)
    #print('reshape(final y shape): ', y.shape)

    # Input 2: generator noise input
    generator_input = Input(shape = (latent_dim,))
    n_nodes = 1024 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer = init)(generator_input)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 1024))(gen)
    #print('Generator noise input: ', gen.shape)
    # Concatenate both the inputs
    merge = Concatenate()([gen, y])
    #print('Concatenate(generator noise input and y: ', merge.shape)

    # (None, 7, 7, 1024) --> (None, 14, 14, 512)
    gen = Conv2DTranspose(512, kernel_size = (5, 5), strides = (2, 2), padding = "same", kernel_initializer = init)(merge)
    gen = BatchNormalization(momentum = 0.9)(gen)
    gen = Activation("relu")(gen)
    #print("(None, 7, 7, 1024) -> (None, 14, 14, 512): ", gen.shape)

    # (None, 14, 14, 512)  --> (None, 28, 28, 256)
    gen = Conv2DTranspose(256, kernel_size = (5, 5), strides = (2, 2), padding = "same", kernel_initializer = init)(gen)
    gen = BatchNormalization(momentum = 0.9)(gen)
    gen = Activation("relu")(gen)
    #print('(None, 14, 14, 512) -> (None, 28, 28, 256): ', gen.shape)

    # (None, 28, 28, 256) --> (None, 56, 56, 128)
    gen = Conv2DTranspose(128, kernel_size = (5, 5), strides = (2, 2), padding = "same", kernel_initializer = init)(gen)
    gen = BatchNormalization(momentum = 0.9)(gen)
    gen = Activation("relu")(gen)
    #print('(None, 28, 28, 256) -> (None, 56, 56, 128): ', gen.shape)

    # (None, 56, 56, 128) --> (None, 112, 112, 3)
    gen = Conv2DTranspose(3, kernel_size = (5, 5), strides = (2, 2), padding = "same", kernel_initializer = init)(gen)
    out_layer = Activation("tanh")(gen)
    #print("(None, 56, 56, 128) -> (None, 112, 112, 3): ", out_layer.shape)
    
    model = Model(inputs = [generator_input, label_input], outputs = out_layer)
    return model
  
g = generator(latent_dim = 100, n_classes = 2)
g.summary()
