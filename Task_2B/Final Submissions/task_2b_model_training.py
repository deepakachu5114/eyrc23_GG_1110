import random
import shutil
import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# os.mkdir("/home/deepakachu/Desktop/eyantra/task2b/validation")
# parent_folder = "/home/deepakachu/Desktop/eyantra/task2b/training"
# for subfolder in os.listdir(parent_folder):
#   os.mkdir(os.path.join("/home/deepakachu/Desktop/eyantra/task2b/validation", subfolder))
# for subfolder in os.listdir(parent_folder):
#   path = os.path.join(parent_folder, subfolder)
#   destination = os.path.join("/home/deepakachu/Desktop/eyantra/task2b/validation", subfolder)
#   files = os.listdir(path)
#   files_to_transfer = random.sample(files, 16)
#   for file_name in files_to_transfer:
#     source_file = os.path.join(path, file_name)
#     shutil.move(source_file, destination)


datagen = ImageDataGenerator()

class_names = ['Combat','Humanitarian Aid and rehabilitation','Fire','Military vehicles and weapons','DestroyedBuildings']

# training data
train_generator = datagen.flow_from_directory(
    directory="/home/deepakachu/Desktop/eyantra/task2b/training",
    classes = class_names,
    target_size=(224, 224),
    batch_size=10,
    class_mode="binary",
)

valid_generator = datagen.flow_from_directory(
    directory="/home/deepakachu/Desktop/eyantra/task2b/validation",
    classes = class_names,
    target_size=(224, 224),
    batch_size=10,
    class_mode="binary",
)

test_generator = datagen.flow_from_directory(
    directory="/home/deepakachu/Desktop/eyantra/task2b/testing",
    classes = class_names,
    target_size=(224, 224),
    batch_size=10,
    class_mode="binary",
)


# ResNet50 model
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in resnet_50.layers:
    layer.trainable = False

def model():
  x = resnet_50.output
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(256, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  predictions = layers.Dense(5, activation='softmax')(x)
  model = Model(inputs = resnet_50.input, outputs = predictions)

  model.summary()
  return model

model = model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# define training function
def trainModel(model, epochs, optimizer):
    batch_size =10
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])


model_history = trainModel(model = model, epochs = 20, optimizer = "Adam")

test_loss, test_acc = model.evaluate(test_generator)
print("The test loss is: ", test_loss)
print("The best accuracy is: ", test_acc*100)
model.save('/home/deepakachu/Desktop/eyantra/task2b/saved_models/model1')

bleh = "/home/deepakachu/Pictures/Screenshots/Screenshot from 2023-10-25 18-21-39.png"
# bleh2 = "/home/deepakachu/Desktop/eyantra/experimetation/images/1.jpg"
def classify(path):
    # class_names = [combat, rehab, fire, military_vehicles,
    #                destroyed_building]

    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions, axis = 1)
    return class_names[class_id.item()]

print(classify(bleh))
# print(classify(bleh2))