#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


names=['sepal-length','sepal-width','petal-length','petal-width','class']
data = pd.read_csv('iris.csv', names=names, skiprows=1)


# In[3]:


print(data[0:10])


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


list(data.columns)


# In[7]:


data.dtypes


# In[9]:


for item in list(data.columns):
    print('----------{}---------'.format(item))
    print(data[item].describe())
    print('\n')


# In[10]:


plt.hist(data['sepal-length'],bins=25,color=['red'])
plt.ylabel('Frequency')
plt.xlabel("Sepal Length")
plt.title("Sepal Length Histogram")
plt.show()


# In[11]:


plt.hist(data['sepal-width'],bins=25,color=['green'])
plt.ylabel('Frequency')
plt.xlabel("Sepal Width")
plt.title("Sepal Width Histogram")
plt.show()


# In[12]:


plt.hist(data['petal-length'],bins=25,color=['orange'])
plt.ylabel('Frequency')
plt.xlabel("Petal Length")
plt.title("Petal Length Histogram")
plt.show()


# In[13]:


plt.hist(data['petal-width'],bins=25,color=['blue'])
plt.ylabel('Frequency')
plt.xlabel("Petal Width")
plt.title("Petal Width Histogram")
plt.show()


# In[14]:


data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize = (8,8), notch=True)
plt.show()

