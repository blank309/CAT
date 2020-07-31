#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv("AI-DataTrain.csv")
data2 = pd.read_csv("AI-DataTest.csv")


# In[4]:


data.head()


# In[5]:


data2.head()


# In[6]:


#data = data.drop('Num',axis=1)


# In[7]:


data.head()


# In[8]:


data = data.values
data2 = data2.values


# In[9]:


[m,n] = data.shape
print(m,n)
[l,p] = data2.shape
print(l,p)


# In[10]:


list1 = []
list2 = []
def prob(data,i,list,row):
    pb = 0
    count = 0
    total = 0
    for j in range(0,row,1):
        if data[j,i]==1:
            count = count + 1
        total = total + 1
    pb = count/total
    list.append(pb)
    
def inter(data,k,list,row):
    for i in range(0,k,1):
        prob(data,i,list,row)
        
inter(data,n,list1,m)
inter(data2,p,list2,l)


# In[11]:


Y_train = []
Y_test = []
def weights(lt,Y,colm):
    for i in range(0,colm,1):
        if lt[i] > 0.700:
            Y.append(2)
        elif lt[i] > 0.350 and lt[i] < 0.699:
            Y.append(1)
        else:
            Y.append(0)
weights(list2,Y_test,p)
weights(list1,Y_train,n)


# In[12]:


X_train = np.array(list1)
X_test = np.array(list2)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_train.shape


# In[13]:


from tensorflow import keras


# In[14]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(12, activation=tf.nn.relu),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


history = model.fit(X_train,Y_train,epochs=2000,validation_data=(X_test,Y_test))


# In[16]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[17]:


classifications = model.predict(X_test)


# In[18]:


testweights = []
testprob = []
for i in range(0,p,1):
    if classifications[i,0] > classifications[i,1] and classifications[i,0] > classifications[i,2]:
        testweights.append('0')
        testprob.append(classifications[i,0])
    elif classifications[i,1] > classifications[i,2]:
        testweights.append('1')
        testprob.append(classifications[i,1])
    else:
        testweights.append('2')
        testprob.append(classifications[i,2])


# In[19]:


Question = []
for i in range(0,p,1):
    k = i+1 
    qs = "Question"+str(k)
    Question.append(qs)


# In[20]:


#pip install openpyxl


# In[22]:


df = pd.DataFrame() 
  
# Creating two columns 
df['Question'] = Question
df['Weights'] = testweights
df['Prob'] = testprob

# Converting to excel 
df.to_excel('result.xlsx', index = False)


# In[ ]:




