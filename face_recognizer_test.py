from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import ExtraTreesRegressor
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from keras import losses
import pickle
import traceback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import RMSprop, Adam, Adadelta
import urllib.request
import io
import numpy as np
import random
import scipy.misc
from keras import optimizers
from PIL import Image
import glob


image_y = 125
image_x = 94

def load():
    lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=1)
    return lfw_people


def create_model():
    nb_filters = 16
    nb_conv = 1

    model = Sequential()

    model.add(Conv2D(16, (2, 2), input_shape=(image_y, image_x, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(16, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.1, decay=1e-5, momentum=0.0, nesterov=False)

    model.compile(loss=losses.categorical_crossentropy, optimizer=sgd, metrics=['mae'])
    return model


if __name__ == '__main__':
    lfw_people = load()
    print(1)