#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobytes : Data Science Internship
# 
# ## Task 5 : To predict how much product will people buy based on factors such as amount spend on advertisement.
# 
# # Sales Prediction Using Python
# 
# ## Author - Patil Saloni Ravindra
# 
# ## March - P2 Batch Oasis Infobyte SIP

# ## Importing Libraries

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataset

# In[7]:


data=pd.read_csv("C:/Users/admin/Documents/Oasis Internship/Advertising.csv")
data


# In[8]:


data.tail()


# In[9]:


data.head()


# In[10]:


data.describe()


# In[11]:


data.shape


# In[12]:


data.isnull().sum()


# In[13]:


data.corr()


# In[14]:


sns.heatmap(data.corr(),cbar=True,linewidths=0.5,annot=True)


# In[15]:


sns.pairplot(data)


# In[16]:


sns.distplot(data['Newspaper'])


# In[17]:


sns.distplot(data['Radio'])


# In[18]:


sns.distplot(data['Sales'])


# In[19]:


sns.distplot(data['TV'])


# ## Data Preprocessing

# In[24]:


data=data.drop(columns=['Unnamed: 0'])


# In[25]:


data


# In[26]:


x=data.drop(['Sales'],1)
x.head()


# In[27]:


y=data['Sales']


# In[28]:


y.head()


# ## Spliting the Dataset

# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)


# In[32]:


print(x.shape,x_train.shape,x_test.shape)


# In[33]:


print(y.shape,y_train.shape,y_test.shape)


# In[34]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
x_test=x_test.astype(int)


# In[38]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# ## Applying Linear Regression

# In[40]:


from sklearn.linear_model import LinearRegression
accuracies={}
lr=LinearRegression()
lr.fit(x_train,y_train)
acc=lr.score(x_test,y_test)*100
accuracies['Linear Regression']=acc
print("Test Accuracy {:.2f}%".format(acc))


# ## Analysing the data by Scatter plot

# In[43]:


y_pred=lr.predict(x_test_scaled)


# In[44]:


plt.scatter(y_test,y_pred,c='r')


# ## Thank You !

# In[ ]:




