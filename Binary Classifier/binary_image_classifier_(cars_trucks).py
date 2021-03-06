# -*- coding: utf-8 -*-
"""Binary Image Classifier (Cars Trucks).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RfNdGG3CNfV-CmIh79hCfRN2YrGP-mmZ
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline

train_path = '/content/drive/My Drive/Colab Notebooks/Eklavya/Cars & Trucks/Train'
valid_path = '/content/drive/My Drive/Colab Notebooks/Eklavya/Cars & Trucks/Valid'
test_path = '/content/drive/My Drive/Colab Notebooks/Eklavya/Cars & Trucks/Test'

train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224),
    classes=['Cars', 'Trucks'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224),
    classes=['Cars', 'Trucks'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224),
    classes=['Cars', 'Trucks'], batch_size=10)

# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)

plots(imgs, titles=labels) # for car: [1,0] & for truck: [0,1]

"""# Build a CNN"""

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(2, activation='softmax'),
    ])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_batches, steps_per_epoch=70, 
    validation_data=valid_batches, validation_steps=8, epochs=10, verbose=2)

"""## Predict"""

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(generator=test_batches, steps=1, verbose=0)
predictions

"""## Confusion Matrix"""

cm = confusion_matrix(y_true=test_labels, y_pred=np.round(predictions[:,0]))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['Cars','Heavy Duty Vehicles']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""## Build fine tuned VGG16 model"""

vgg16_model = keras.applications.vgg16.VGG16()

vgg16_model.summary()

type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

type(model)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.summary()

"""## Train a fine tuned model"""

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, steps_per_epoch = 70, validation_data=valid_batches, validation_steps = 8, epochs=10, verbose=2)

"""## Predict using fine tuned VGG16 model"""

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles = test_labels)

test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(test_batches, steps=8, verbose=0)
#predictions = model.predict(x=test_batches, verbose=0)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['Cars','Heavy Duty Vehicles']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

