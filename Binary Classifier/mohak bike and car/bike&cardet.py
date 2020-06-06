#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
#import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 
import os
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


data_path=r"C:\Users\MOHAK\Desktop\carbike\training"
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels)) 


# In[7]:


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
        print("Exception",e)
print(data)


# In[8]:


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


# In[9]:


from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(train_data[0], train_data[1], test_size=0.25, random_state=42)


# In[10]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(264, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[11]:


history=model.fit(trainX, trainY, batch_size = 25, epochs = 10, verbose = 1, validation_data = (testX, testY)) 


# In[12]:


import cv2,os

data_path1=r'C:\Users\MOHAK\Desktop\carbike\testing'
categories1=os.listdir(data_path1)
labels1=[i for i in range(len(categories1))]

label_dict1=dict(zip(categories1,labels1))


# In[13]:


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


# In[14]:


import numpy as np

data1=np.array(data1)
data1=data1/255
data1=np.reshape(data1,(data1.shape[0],img_size1,img_size1,1))
target1=np.array(target1)

from keras.utils import np_utils

new_target1=np_utils.to_categorical(target1)


# In[15]:


model.evaluate(data1,new_target1)


# In[16]:


classes=model.predict(data1)
print(classes)


# In[19]:


model.save('carbike2.h5')


# In[ ]:





# In[ ]:




