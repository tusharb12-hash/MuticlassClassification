#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


# In[14]:


data=[]

testpath=r"C:\Users\MOHAK\Desktop\test"
imgs=os.listdir(testpath)
for img in imgs:
    print(img)
    impath=os.path.join(testpath, img)
    pict=cv2.imread(impath)
    gray=cv2.cvtColor(pict, cv2.COLOR_BGR2GRAY)
    pic=cv2.resize(gray,(64,64))
    data.append(pic)
    
data=np.array(data)
data=data/255
data=np.reshape(data,(data.shape[0],64,64,1))


classify = load_model(r"C:\Users\MOHAK\buscarbike4.h5")

classify.summary()

y = classify.predict(data)
for elements in y:
    if elements[0] > 0.5:
        print('bike')
    elif elements[1]>0.5:
        print('bus')
    elif elements[2]>0.5:
        print('car')
    else:
        print("none")


# In[ ]:




