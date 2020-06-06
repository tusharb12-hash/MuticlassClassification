# -*- coding: utf-8 -*-
"""bus&car.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sk6qInjRptv2J9Y2MQ48MRRQUsDyOFyJ
"""

# Creating data

import os

dataset_path = '/content/drive/My Drive/Cars or Bus'

categories = os.listdir(dataset_path)
label = [i for i in range(len(categories))]

print(categories)
print(label)

label_dict = dict(zip(categories, label))   # { 'bus' : 0, 'cars' : 1 }
print(label_dict)

import cv2
from google.colab.patches import cv2_imshow

data = []
target = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)

    # print(folder_path)
    for images in os.listdir(folder_path):
        image_path = os.path.join(folder_path, images)

        # print(image_path)
        try:
            img = cv2.imread(image_path)
            
            resize = cv2.resize(img, (64, 64))

            data.append(resize)
            target.append(label_dict[category])

        except Exception as e:
            print(e)

import numpy as np

data = np.array(data, dtype = 'float32') / 255
data = data.reshape(data.shape[0], 64, 64, 3)
target = np.array(target)

# data
from keras.utils import np_utils
new_target = np_utils.to_categorical(target)

# new_target

from sklearn.utils import shuffle
data, Label = shuffle(data, new_target, random_state = 42)

TrainTestData = [data, Label]
# Label

from sklearn.model_selection import train_test_split

(trainX, testX, trainY, testY) = train_test_split(data, Label, test_size = 0.25, random_state = 2)

# print(len(trainX), len(testX), len(trainY), len(testY))

from keras.preprocessing.image import ImageDataGenerator

augment = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(x = augment.flow(trainX, trainY, batch_size = 32), 
                    validation_data = (testX, testY),
                    steps_per_epoch = 1914//32 + 100,
                    epochs = 10,
                    verbose = 1
)

img1 = cv2.imread('/content/drive/My Drive/bus.jpg')
img2 = cv2.imread('/content/drive/My Drive/car.jpg')
img3 = cv2.imread('/content/drive/My Drive/bus1.jpg')
img4 = cv2.imread('/content/drive/My Drive/road.jpg')


img1 = cv2.resize(img1, (64, 64))
img2 = cv2.resize(img2, (64, 64))
img3 = cv2.resize(img3, (64, 64))
img4 = cv2.resize(img4, (64, 64))


test = []
test.append(img1)
test.append(img2)
test.append(img3)
test.append(img4)

cv2_imshow(img1)
cv2_imshow(img2)
cv2_imshow(img3)
cv2_imshow(img4)

test = np.array(test, dtype = 'float32') / 255
test = test.reshape(test.shape[0], 64, 64, 3)

model.predict(test)

model.evaluate(testX, testY)

model.save('carbus.h5')

datagen = ImageDataGenerator(rescale = 1./255)
testing = datagen.flow_from_directory(
    directory = '/content/drive/My Drive/Test',
    target_size = (64, 64),
    classes = ['Bus', 'Car'],
    shuffle = True,
)

from keras.models import load_model

classify = load_model('/content/carbus.h5')
classify.summary()

classify.evaluate_generator(testing, verbose = 1)