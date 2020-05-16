#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[24]:


plt.gray() 
for i in range(2):
    plt.matshow(digits.images[i])


# In[25]:


dir(digits)


# In[26]:


digits.data[0]


# In[27]:


digits.images[4]


# In[28]:


plt.matshow(digits.images[9])


# In[29]:


digits.target[0:7]


# In[30]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


# In[33]:


model.fit(X_train, y_train)


# In[34]:


model.score(X_test, y_test)


# In[35]:


model.predict(digits.data[0:5])


# In[36]:


y_predicted = model.predict(X_test)


# In[37]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[38]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[44]:


model.predict([digits.data[6]])


# In[45]:


digits.target[5]


# In[46]:


model.predict([digits.data[5]])


# In[47]:


digits.target[6]


# In[48]:


model.predict([digits.data[6]])


# In[ ]:




