#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


filename = './data/Q1/MNIST_Subset.h5'
f = h5py.File(filename, 'r')
X = np.array(f[list(f.keys())[0]])
y = np.array(f[list(f.keys())[1]])
print('Number of samples in the dataset: ', len(y))

unique, counts = np.unique(y, return_counts=True)
print('Distribution of samples among different classes: ', counts)
print(X.shape)


# In[3]:


X = np.vstack([img.reshape(-1, ) for img in X])
print(X.shape)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[5]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1.0 - z)


# In[6]:





# In[6]:





# In[10]:


def train(X, y, n_nodes):
    m = X.shape[0]
    # n_layers = len(n_nodes)
    lr = 0.1

    y = np.reshape(y, (len(y), 1))

    w1 = np.zeros((len(X[0]), 100))  # input dim x nodes
    b1 = np.zeros((1, 100))

    w2 = np.zeros((100, 1))  # input dim x nodes
    b2 = np.zeros((1, 1))

    errors = []

    for i in (range(100)):
        # Forward Propagation
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        # print('Layer 1 output shape:', a1.shape)

        z2 = np.dot(a1, w2) + b2
        exp_scores = np.exp(z2)
        # print('here1')
        
        # a2 = 1
        a2 = exp_scores
        # print(a2.shape)
        exp_sum = np.sum(exp_scores, axis=1, keepdims=True) + 1
        # break
        # print(exp_sum)
        a2 = exp_scores/exp_sum

        # print(a2)
        
        # break
        
        # print('Layer 2 output shape:', a2.shape)

        # if len(np.where(a1 == 0)) == 0 and len(np.where(a1 == 1)) == 0:
        #     cost = np.sum(y * np.log(a2) + (1 - y) * (np.log(1 - a2)))
        #     if i % 100:
        #         print(cost, end=', ')

        # print('here2')

        # Backward Propagation
        error = y - a2

        # print('here 3')

        dw2 = np.dot(a1.T, error)
        db2 = np.sum(error)

        # print('here 4')

        delta = np.dot(error, w2.T) * sigmoid_prime(a1)
        dw1 = np.dot(X.T, delta)
        db1 = np.sum(delta)

        # print('here 5')

        # print(error)
        error = np.mean(np.abs(error), axis=0)
        errors.append(error)

        # Update Weights
        w2 += -lr * dw2
        b2 += -lr * db2
        w1 += -lr * dw1
        b1 += -lr * db1

    return w1, b1, w2, b2, errors


# In[11]:


w1, b1, w2, b2, errors = train(X_train, y_train, 0)
accuracy = (1 - errors[-1]) * 100


# In[117]:


# print("Training Accuracy " + str(round(accuracy, 2)) + "%")
# print(errors)
# plt.plot(errors)
# plt.xlabel('Training')
# plt.ylabel('Error')
# plt.show()


# In[ ]:




