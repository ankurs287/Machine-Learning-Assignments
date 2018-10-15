
# coding: utf-8

# # Logistic Regression

# In[1]:


import pickle

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


# ## Import Data

# In[3]:


train_images = idx2numpy.convert_from_file('./data/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('./data/train-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./data/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./data/t10k-labels-idx1-ubyte')


# ### reshape images 3D vector to 2D vector 
# 

# In[4]:


X_train = np.vstack([img.reshape(-1, ) for img in train_images])
y_train = train_labels
X_test = np.vstack([img.reshape(-1, ) for img in test_images])
y_test = test_labels
X_train, y_train = shuffle(X_train, y_train)
train_cap = 5000
X_train = X_train[:train_cap, :]
y_train = y_train[:train_cap]
dataset_size = len(y_train)
print(dataset_size)


# In[13]:


logit_l2_model_file = './saved_model/q2_12.sav'
logit_l1_model_file = './saved_model/q2_l1.sav'


# In[14]:


from sklearn.linear_model import LogisticRegression


# ### L2 Norm
# 

# In[15]:


def logistic_regression(penalty='l2'):
    try:
        loaded_model = pickle.load(open(logit_l2_model_file if penalty == 'l2' else logit_l1_model_file, 'rb'))
        print('local model returned')
        return loaded_model
    except FileNotFoundError:
        regs = []
    for i in range(10):
        print('training digit ..', i)
        y_train_ = y_train.copy()
        y_train_ = (y_train_ == i).astype(int)
        reg = LogisticRegression(penalty=penalty)
        reg.fit(X_train, y_train_)
        regs.append(reg)
    pickle.dump(regs, open(logit_l2_model_file if penalty == 'l2' else logit_l1_model_file, 'wb'))
    print('Fresh model returned')
    return regs


# In[16]:


reg_l2 = logistic_regression()


# In[17]:


for i in range(10):
    y_train_ = y_train.copy()
    y_train_ = (y_train_ == i).astype(int)

    train_accuracy = reg_l2[i].score(X_train, y_train_)
    print('Train Accuracy for %sth digit: %s' % (i, train_accuracy))

    y_test_ = y_test.copy()
    y_test_ = (y_test_ == i).astype(int)
    test_accuracy = reg_l2[i].score(X_test, y_test_)
    print('Test Accuracy for %sth digit: %s\n' % (i, test_accuracy))


# ### L1 Norm

# In[18]:


reg_l1 = logistic_regression(penalty='l1')


# In[19]:


for i in range(10):
    y_train_ = y_train.copy()
    y_train_ = (y_train_ == i).astype(int)

    train_accuracy = reg_l1[i].score(X_train, y_train_)
    print('Train Accuracy for %sth digit: %s' % (i, train_accuracy))

    y_test_ = y_test.copy()
    y_test_ = (y_test_ == i).astype(int)
    test_accuracy = reg_l1[i].score(X_test, y_test_)
    print('Test Accuracy for %sth digit: %s\n' % (i, test_accuracy))

