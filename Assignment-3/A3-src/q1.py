
# coding: utf-8

# In[1]:


import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
from sklearn.svm import SVC
from copy import deepcopy

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset

# In[2]:


filename = './data/Q1/MNIST_Subset.h5'
f = h5py.File(filename, 'r')
X = np.array(f[list(f.keys())[0]])
y = np.array(f[list(f.keys())[1]])
print('Number of samples in the dataset: ', len(y))

unique, counts = np.unique(y, return_counts=True)
print('Distribution of samples among different classes: ', counts)
print(X.shape)


# ## Preprocessing

# In[3]:


from sklearn.preprocessing import MinMaxScaler

X = np.vstack([img.reshape(-1, ) for img in X])
print(X.shape)
X = MinMaxScaler().fit_transform(X)
y = np.where(y == 7, 0, 1)


# ## Train Test Split

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Define Activation Function for Hidden Layers

# In[5]:


def sigmoid(z, deriv=False):
    if deriv:
        return z * (1.0 - z)
    return 1 / (1 + np.exp(-z))


def relu(z, deriv=False):
    if deriv:
        z[z<=0] = 0
        z[z>0] = 1
        return z
    return z * (z > 0)


# ## General Neural Network-
# Hidden Layers with Sigmoid/ Relu Activation Function.  
# Output Layer with Softmax Activation Function.

# In[6]:


class Layer:
    def __init__(self, input_dim, n_nodes, activation=sigmoid):
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.w = 2 * np.random.random((input_dim, n_nodes)) - 1  # input dim x nodes
        self.b = 2 * np.random.random((1, n_nodes)) - 1
        self.activation =  activation

class NeuralNetwork:
    def __init__(self, layers, lr=1):
        self.layers = layers
        self.lr = lr

    def build_train_model(self, X, y, X_test, y_test, n_iterations):
        m = X.shape[0]
        l = X_test.shape[0]
        n_layers = len(self.layers)

        lr = self.lr

        y = np.reshape(y, (len(y), 1))
        y_ = np.where(y == 0, 1, 0)
        y = np.hstack((y, y_))

        y_test = np.reshape(y_test, (len(y_test), 1))
        y_test_ = np.where(y_test == 0, 1, 0)
        y_test = np.hstack((y_test, y_test_))
        
        train_errors = []
        val_errors = []

        for i in tqdm_notebook(range(n_iterations)):
            inp = X
            # Forward Propagation
            y_pred_val = None
            for j in range(n_layers):
                layer = self.layers[j]
                layer.inp = inp

                if j == n_layers - 1:
                    z = np.dot(inp, layer.w) + layer.b
                    exp_scores = np.exp(z)
                    exp_sum = np.sum(exp_scores, axis=1, keepdims=True) + 0.00001
                    layer.a = exp_scores / exp_sum
                else:
                    z = np.dot(inp, layer.w) + layer.b
                    layer.a = layer.activation(z)

                inp = layer.a

            error = y - self.layers[-1].a
            
#             J = -1*(1/m)*np.sum((np.log(self.layers[-1].a)*(y) + np.log(1-self.layers[-1].a)*(1-y)))
            y_pred_val = self.predict(X_test)
#             J_val = -1*(1/l)*np.sum((np.log(y_pred_val)*(y_test) + np.log(1-y_pred_val)*(1-y_test)))
#             val_cost.append(J_val)
#             cost.append(J)
            
            error_ = np.mean(np.abs(error[:, 0]))
            train_errors.append(error_)
            verror_ = np.mean(np.abs((y_test - y_pred_val)[:, 0]))
            val_errors.append(verror_)
            
            if i % 200 == 0:
                print('Train Error: ', np.round(error_, 4), 'Val Cost: ', np.round(verror_, 4))

            # Backward Propagation
            for j in range(n_layers - 1, -1, -1):
                layer = self.layers[j]
                if j == n_layers - 1:
                    layer.dw = np.dot(self.layers[j - 1].a.T, error)
                    layer.db = np.sum(error)
                else:
                    delta = np.dot(error, self.layers[j + 1].w.T) * layer.activation(layer.a, deriv=True)
                    layer.dw = np.dot(layer.inp.T, delta)
                    layer.db = np.sum(delta)
                    error = delta

            # Update Weights
            for j in range(n_layers):
                layer = self.layers[j]
                layer.w += 1 / m * lr * layer.dw
                layer.b += 1 / m * lr * layer.db

        return train_errors, val_errors
    
    def predict(self, X_test):
        n_layers = len(self.layers)
        layers = deepcopy(self.layers)
        inp = X_test
        for j in range(n_layers):
            layer = layers[j]
            if j == n_layers - 1:
                z = np.dot(inp, layer.w) + layer.b
                exp_scores = np.exp(z)
                exp_sum = np.sum(exp_scores, axis=1, keepdims=True)
                layer.a = exp_scores / exp_sum
            else:
                z = np.dot(inp, layer.w) + layer.b
                layer.a = layer.activation(z)

            inp = layer.a

        return layers[-1].a


# ## Train and Test NN with 1 hidden layer `Sigmoid`

# In[7]:


layers = []
h1 = Layer(len(X_train[0]), 100)
layers.append(h1)
o = Layer(100, 2)
layers.append(o)


# In[8]:


nn = NeuralNetwork(layers, lr=0.9)
train_errors, val_errors = nn.build_train_model(X_train, y_train, X_test, y_test, 2000)


# In[9]:


accuracy = 1 - train_errors[-1]
accuracy = np.round(accuracy, 4) * 100
print("Training Accuracy " + str(accuracy) + "%")
plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('1 Hidden Layer with Sigmoid')
plt.legend()
plt.savefig('1 Hidden Layer with Sigmoid.png')
plt.show()


# In[10]:


y_pred = nn.predict(X_test)
error = y_test - y_pred[:, 0]
error = np.mean(np.abs(error))
accuracy = 1 - error
accuracy = np.round(accuracy, 4) * 100
print("Test Accuracy " + str(accuracy) + "%")


# In[11]:


# Store data (serialize)
with open('sigmoid_1_layer_mlp.pickle', 'wb') as handle:
    pickle.dump(nn, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Train and Test NN with 3 hidden layer `Sigmoid`

# In[13]:


layers = []
h1 = Layer(len(X_train[0]), 100)
layers.append(h1)
h2 = Layer(100, 50)
layers.append(h2)
h3 = Layer(50, 50)
layers.append(h3)
o = Layer(50, 2)
layers.append(o)


# In[14]:


nn = NeuralNetwork(layers, lr=0.9)
train_errors, val_errors = nn.build_train_model(X_train, y_train, X_test, y_test, 2000)


# In[15]:


accuracy = 1 - train_errors[-1]
accuracy = np.round(accuracy, 4) * 100
print("Training Accuracy " + str(accuracy) + "%")
plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('3 Hidden Layer with Sigmoid')
plt.legend()
plt.savefig('3 Hidden Layer with Sigmoid.png')
plt.show()


# In[16]:


y_pred = nn.predict(X_test)
error = y_test - y_pred[:, 0]
error = np.mean(np.abs(error))
accuracy = 1 - error
accuracy = np.round(accuracy, 4) * 100
print("Test Accuracy " + str(accuracy) + "%")


# In[17]:


# Store data (serialize)
with open('sigmoid_3_layer_mlp.pickle', 'wb') as handle:
    pickle.dump(nn, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Train and Test NN with 1 hidden layer `Relu`

# In[22]:


layers = []
h1 = Layer(len(X_train[0]), 100, activation=relu)
layers.append(h1)
o = Layer(100, 2)
layers.append(o)


# In[23]:


nn = NeuralNetwork(layers, lr=0.1)
train_errors, val_errors = nn.build_train_model(X_train, y_train, X_test, y_test, 2000)


# In[24]:


accuracy = 1 - train_errors[-1]
accuracy = np.round(accuracy, 4) * 100
print("Training Accuracy " + str(accuracy) + "%")
plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('1 Hidden Layer with Relu')
plt.legend()
plt.savefig('1 Hidden Layer with Relu.png')
plt.show()


# In[25]:


y_pred = nn.predict(X_test)
error = y_test - y_pred[:, 0]
error = np.mean(np.abs(error))
accuracy = 1 - error
accuracy = np.round(accuracy, 4) * 100
print("Test Accuracy " + str(accuracy) + "%")


# In[26]:


# Store data (serialize)
with open('relu_1_layer_mlp.pickle', 'wb') as handle:
    pickle.dump(nn, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Train and Test NN with 3 hidden layer `Relu`

# In[32]:


layers = []
h1 = Layer(len(X_train[0]), 100,  activation=relu)
layers.append(h1)
h2 = Layer(100, 50,  activation=relu)
layers.append(h2)
h3 = Layer(50, 50,  activation=relu)
layers.append(h3)
o = Layer(50, 2)
layers.append(o)


# In[33]:


nn = NeuralNetwork(layers, lr=0.001)
train_errors, val_errors = nn.build_train_model(X_train, y_train, X_test, y_test, 2000)


# In[34]:


accuracy = 1 - train_errors[-1]
accuracy = np.round(accuracy, 4) * 100
print("Training Accuracy " + str(accuracy) + "%")
plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('3 Hidden Layer with Relu')
plt.legend()
plt.savefig('3 Hidden Layer with Relu.png')
plt.show()


# In[35]:


y_pred = nn.predict(X_test)
error = y_test - y_pred[:, 0]
error = np.mean(np.abs(error))
accuracy = 1 - error
accuracy = np.round(accuracy, 4) * 100
print("Test Accuracy " + str(accuracy) + "%")


# In[39]:


# Store data (serialize)
with open('relu_3_layer_mlp.pickle', 'wb') as handle:
    pickle.dump(nn, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## SVM

# In[36]:


clf = SVC(verbose=True)
clf.fit(X_train, y_train)


# In[40]:


accuracy = clf.score(X_train, y_train)
print("Training Accuracy " + str(accuracy*100) + "%")


# In[41]:


accuracy = clf.score(X_test, y_test)
print("Test Accuracy " + str(accuracy*100) + "%")

