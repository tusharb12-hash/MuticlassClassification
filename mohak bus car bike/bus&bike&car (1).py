#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np 
#import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 
import os
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


# In[92]:


data_path=r"C:\Users\MOHAK\Desktop\buscarbike\training"
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels)) 


# In[93]:


img_size=64
data=[]
target=[]


for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    
    for img_name in img_names :
      path=os.path.join(folder_path,img_name)
      img=cv2.imread(path)

      try:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        resized=cv2.resize(gray,(img_size,img_size))

        data.append(resized)
        target.append(label_dict[category])

      except Exception as e:
        print(img_name)
        print("Exception",e)


# In[94]:


import numpy as np

data=np.array(data)
data=data/255
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)

from sklearn.utils import shuffle
data,Label = shuffle(data,new_target, random_state=2)
train_data = [data,Label]


# In[95]:


from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(train_data[0], train_data[1], test_size=0.25, random_state=42)


# In[96]:


augment=ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# In[119]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
#kernel_regularizer=regularizers.l2(0.1)
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()


# In[120]:


history=model.fit( x=augment.flow(trainX, trainY, batch_size = 25),validation_data = (testX, testY),steps_per_epoch=400,  epochs = 20, verbose = 1) 


# In[99]:


import cv2,os

data_path1=r'C:\Users\MOHAK\Desktop\buscarbike\testing'
categories1=os.listdir(data_path1)
labels1=[i for i in range(len(categories1))]
label_dict1=dict(zip(categories1,labels1))


# In[100]:


img_size1=64
data1=[]
target1=[]


for category1 in categories1:
    folder_path1=os.path.join(data_path1,category1)
    img_names1=os.listdir(folder_path1)
    
    for img_name1 in img_names1 :
      path1=os.path.join(folder_path1,img_name1)
      img1=cv2.imread(path1)

      try:
        gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        resized1=cv2.resize(gray1,(img_size1,img_size1))

        data1.append(resized1)
        target1.append(label_dict1[category1])

      except Exception as e:
        print("Exception",e)


# In[101]:


import numpy as np

data1=np.array(data1)
data1=data1/255
data1=np.reshape(data1,(data1.shape[0],img_size1,img_size1,1))
target1=np.array(target1)

from keras.utils import np_utils

new_target1=np_utils.to_categorical(target1)


# In[121]:


model.evaluate(data1,new_target1)


# In[109]:


classes=model.predict(data1)
print(classes)


# In[118]:


#model.save('buscarbike5.h5')

