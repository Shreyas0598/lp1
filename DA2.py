#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as no
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read data from csv file
headers = ['Pregnancy Count', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age', 'Has Diabetes']
data=pd.read_csv("Pima.csv",names=headers)


# In[3]:


# Print the dimensions of the dataset
data.shape


# In[4]:


# Describe the dataset
data.describe()


# In[5]:


# Plot histograms of each feature
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[6]:


# Seperate the dataset into training and testing set (80% training and 20% testing)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
trainingSet,testingSet=train_test_split(data,test_size=0.20,random_state=0)


# In[7]:


# Remove the outcome column from the dataset
trainingSetLabels=trainingSet["Has Diabetes"].copy()
trainingSet=trainingSet.drop("Has Diabetes", axis=1)

testingSetLabels=testingSet["Has Diabetes"].copy()
testingSet=testingSet.drop("Has Diabetes",axis=1)


# In[8]:


# Dimensions of training set
trainingSet.shape


# In[9]:


# Dimensions of testing set
testingSet.shape


# In[10]:


# Initialize classifier and train it on the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB();
classifier.fit(trainingSet,trainingSetLabels)


# In[11]:


# Predict the result of testing data
predictedValues=classifier.predict(testingSet);
len(predictedValues)


# In[12]:


# Calculate the accuracy of the predicted values
accuracy=accuracy_score(testingSetLabels,predictedValues)
print(accuracy)


# In[13]:


cm=confusion_matrix(testingSetLabels,predictedValues)


# In[15]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(testingSetLabels,predictedValues)


# In[16]:


cm


# In[17]:


import seaborn as sn
sn.heatmap(cm, annot=True)


# In[ ]:




