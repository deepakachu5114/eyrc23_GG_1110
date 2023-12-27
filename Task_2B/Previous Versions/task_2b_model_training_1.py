from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

import scipy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LeakyReLU, Activation
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import VGG16
import random
from tensorflow.keras import layers
import tensorflow as tf

train_path = r'C:\Users\AISHINI\PycharmProjects\pythonProject6\Training'
'''for subfolder in os.listdir(train_path):
  path = os.path.join(train_path, subfolder)
  destination = ("C:\\Users\\AISHINI\\PycharmProjects\\pythonProject6\\Validation\\"+subfolder)
  files = os.listdir(path)
  files_to_transfer = random.sample(files, 10)
  for file_name in files_to_transfer:
    source_file = os.path.join(path, file_name)
    shutil.move(source_file, destination)'''

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

def preprocess(train_path, validation_path):
  train_datagen = ImageDataGenerator(rescale = 1./255)
  train_images = train_datagen.flow_from_directory(train_path, batch_size = 32, target_size = (200, 200))

  validation_datagen = ImageDataGenerator(rescale = 1./255)
  validation_images = validation_datagen.flow_from_directory(validation_path, batch_size = 32, target_size = (200, 200))

  return train_images, validation_images

BatchNormalization(momentum=0.9)

pre_train = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
for layer in pre_train.layers:
    layer.trainable = False

def model():
  input_layer = Input((200, 200, 3))
  x = data_augmentation(input_layer)
  x = pre_train(x)
  x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)

  x = Flatten()(x)

  x = Dense(128)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Dropout(rate = 0.5)(x)

  x = Dense(5)(x)
  output_layer = Activation('softmax')(x)

  model = Model(input_layer, output_layer)

  model.summary()
  return model

from tensorflow.keras.models import load_model

def train():
  validation_path = r'C:\Users\AISHINI\PycharmProjects\pythonProject6\Validation'
  train_images, validation_images = preprocess(train_path, validation_path)
  Model = model()
  opt = Adam(learning_rate=0.0005)
  Model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
  trained = Model.fit(train_images
            , epochs = 15
            , batch_size = 32
            , validation_data = validation_images
            , callbacks = [checkpoint])
  Model.evaluate(validation_images, batch_size=32)
  Model.save(r'C:\Users\AISHINI\PycharmProjects\pythonProject6\my_model10.h5')

  return trained

train()
