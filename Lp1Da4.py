#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('store.csv')


# In[ ]:


data.head()   ######info about data


# In[ ]:


data.info()    ###### info about data


# In[ ]:


data.dtypes   ##### info about types


# In[ ]:


data['Duration'].describe()  #### statistics


# In[ ]:


data=data.drop('Start date',axis=1)
data=data.drop('End date',axis=1)
data=data.drop('Start station',axis=1)
data=data.drop('End station',axis=1)


# In[ ]:


data.head()                                    ########### label encoder
le = LabelEncoder()
le.fit(data['Member type'])
data['Member type'] = le.transform(data['Member type'])


# In[ ]:


le = LabelEncoder()
le.fit(data['Bike number'])
data['Bike number'] = le.transform(data['Bike number'])


# In[ ]:


data.head()


# In[ ]:


data.shape     ###### data.size


# In[ ]:


train=np.array(data.iloc[0:85000])   ### spitting into training and tetsign
test=np.array(data.iloc[85000:,])


# In[ ]:


train.shape,test.shape       ########  train and test


# In[ ]:


from sklearn.naive_bayes import GaussianNB   ##### guassinan
model=GaussianNB()


# In[ ]:


model.fit(train[:,0:4],train[:,4])
predicted=model.predict(test[:,0:4])


# In[ ]:


predicted.shape


# In[ ]:


predicted


# In[ ]:


count=0                 ### accuracy
for l in range(30597):  
    if(predicted[l]==test[l,4]):
        count=count+1


# In[ ]:


count


# In[ ]:


print(count/30597)

