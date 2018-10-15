
# coding: utf-8

# # 1. Linear Regression

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# ### Import dataset

# In[7]:


data = pd.read_excel("./data/boston.xls")
data_set_size = len(data)
data.head()


# ### Preprocessing: Feature Normalisation
# Extract features and normalise them

# In[8]:


def feature_normalisation(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mu) / std


# In[9]:


normalised_data = feature_normalisation(data.values)

np.random.shuffle(normalised_data)

X = normalised_data[:, :-1]
y = normalised_data[:, -1]
print(data_set_size)


# ## Gradient Descent

# In[10]:


def gradient_descent(X, y, alpha=0.01, time=1000, lam=None, reg='ridge'):
    # Adding intercept
    X = np.hstack((np.ones((len(X), 1)), X))
    n = len(X)
    theta = np.zeros((time, len(X[0])))
    train_rmse = np.zeros(time)
    for i in range(time):
        pred = np.dot(X, theta[i - 1])
        error = np.subtract(pred, y)
        derivative = (alpha / n) * np.dot(X.T, error)
        if lam is not None:
            if reg == 'ridge':
                derivative = np.add(derivative, (lam / n) * theta[i - 1])
            elif reg == 'lasso':
                for d in range(len(derivative)):
                    derivative[d] += (lam / n) * (1.0 if theta[i - 1][d] >= 0 else -1.0)

        theta[i] = theta[i - 1] - derivative  # Gradient Descent Rule

        new_pred = np.dot(X, theta[i])
        new_error = np.subtract(new_pred, y)
        train_rmse[i] = np.sqrt((1. / n) * np.dot(new_error.T, new_error))
    return theta, train_rmse


# ## Least Squared Regression [ R(w) = 0 ] for 5-Folds

# In[11]:


time = 1000
k = 5  # number of folds
theta = [None] * k
train_rmse_collection = np.zeros((k, time))
val_rmse_collection = np.zeros((k, time))

kf = KFold(n_splits=k)
i = 0
plt.figure(figsize=(10, 60))
folds = []
for train, val in kf.split(X):
    folds.append([train, val])
    X_fold = X[train]
    y_fold = y[train]
    theta[i], train_rmse = gradient_descent(X_fold, y_fold, time=time)
    0

    # print(train_rmse[-1])

    # Calculate rmse of validation set
    validation_size = len(X[val])
    X_val = X[val]
    y_val = y[val]
    # Adding intercept
    X_val = np.hstack((np.ones((len(X_val), 1)), X_val))
    val_rmse = np.zeros(time)
    for j in range(time):
        pred = np.dot(X_val, theta[i][j])
        error = np.subtract(pred, y_val)
        val_rmse[j] = np.sqrt((1. / validation_size) * np.dot(error.T, error))

    plt.figure()
    # plt.subplot(7, 1, i + 1)
    plt.plot(train_rmse, label='Train RMSE for Fold: %s' % (i + 1))
    plt.plot(val_rmse, label='Validation RMSE for Fold: %s' % (i + 1))
    plt.title('RMSE for Fold: %s' % (i + 1))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.savefig('./plots/q1_a %s.png' % (i + 1), bbox_inches='tight', pad_inches=0.5)
    plt.close()

    train_rmse_collection[i] = train_rmse
    val_rmse_collection[i] = val_rmse

    i += 1

plt.figure()
# plt.subplot(7, 1, 6)
plt.plot(np.mean(train_rmse_collection, axis=0), label='Train Mean RMSE over all Folds')
plt.plot(np.mean(val_rmse_collection, axis=0), label='Validation Mean RMSE over all Folds')
plt.title('Mean RMSE over all Folds')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean RMSE')
plt.savefig("./plots/q1_a 6.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

plt.figure()
# plt.subplot(7, 1, 7)
plt.plot(np.std(train_rmse_collection, axis=0), label='Train Standard Deviation RMSE over all Folds')
plt.plot(np.std(val_rmse_collection, axis=0), label='Validation Standard Deviation RMSE over all Folds')
plt.title('Standard Deviation RMSE over all Folds')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('STD RMSE')
plt.savefig("./plots/q1_a 7.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

# plt.savefig("./q1_a.png", bbox_inches='tight', pad_inches=0.5)
# plt.close()


# In[12]:


for i in range(k):
    print('Train RMSE for Fold: %s : %s' % (i + 1, train_rmse_collection[i, -1]))
    print('Validation RMSE for Fold: %s : %s' % (i + 1, val_rmse_collection[i, -1]))
    print()

train_mean_rmse = np.mean(train_rmse_collection, axis=0)
val_mean_rmse = np.mean(val_rmse_collection, axis=0)
print('Train Mean RMSE over all Folds %s' % (train_mean_rmse[-1]))
print('Validation Mean RMSE over all Folds %s' % (val_mean_rmse[-1]))
print()
train_std_rmse = np.std(train_rmse_collection, axis=0)
val_std_rmse = np.std(val_rmse_collection, axis=0)
print('Train STD RMSE over all Folds %s' % (train_std_rmse[-1]))
print('Validation STD RMSE over all Folds %s' % (val_std_rmse[-1]))


# ## Regularization

# In[13]:


# Find the fold with lowest RMSE for Validation Set and take it out as a test set
test_fold = np.argmin(val_rmse_collection[:, -1])
X_test = X[folds[test_fold][1]]
y_test = y[folds[test_fold][1]]

X_ = np.delete(X, folds[test_fold][1], axis=0)
y_ = np.delete(y, folds[test_fold][1], axis=0)
print('Fold with minimum RMSE on Validation Set: %s (RMSE: %s)' %(test_fold+1, val_rmse_collection[test_fold, -1]))
print('Dataset Size: ', len(y), '\nTrain+Val size: ', len(y_), '\nTest Size: ', len(y_test))


# ## Ridge Regression [ l2 norm ]

# In[14]:


parameters = {'alpha': [0.001, 0.01, 0.1]}
reg = GridSearchCV(Ridge(), parameters, cv=k)
reg.fit(X_, y_)
l = reg.best_params_['alpha']  # Regularisation Parameter
print(l)


# In[15]:


theta, l2_rmse = gradient_descent(X_, y_, time=time, lam=l)

# Calculate rmse of test set
test_size = len(X_test)
# Adding intercept
X_test_ = np.hstack((np.ones((len(X_test), 1)), X_test))
l2_test_rmse = np.zeros(time)
for i in range(time):
    pred = np.dot(X_test_, theta[i])
    error = np.subtract(pred, y_test)
    l2_test_rmse[i] = np.sqrt((1. / test_size) * np.dot(error.T, error))


# In[16]:


# plt.figure(figsize=(10, 20))
# plt.subplot(2, 1, 1)
plt.figure()
plt.plot(l2_rmse, label='Train')
plt.plot(l2_test_rmse, label='Test')
plt.title('L2 Regularization')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.savefig("./plots/q1_reg_l2.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

print('RMSE on Train Set: %s' % (l2_rmse[-1]))
print('RMSE on Test Set: %s' % (l2_test_rmse[-1]))


# ## LASSO Regression [ l1 norm ]

# In[17]:


parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}
reg = GridSearchCV(Lasso(), parameters, cv=k)
reg.fit(X_, y_)
l = reg.best_params_['alpha']  # Regularisation Parameter
print(l)


# In[18]:


theta, l1_rmse = gradient_descent(X_, y_, time=time, lam=l, reg='lasso')

# Calculate rmse of test set
test_size = len(X_test)
# Adding intercept
X_test_ = np.hstack((np.ones((len(X_test), 1)), X_test))
l1_test_rmse = np.zeros(time)
for i in range(time):
    pred = np.dot(X_test_, theta[i])
    error = np.subtract(pred, y_test)
    l1_test_rmse[i] = np.sqrt((1. / test_size) * np.dot(error.T, error))


# In[19]:


# plt.subplot(2, 1, 2)
plt.figure()
plt.plot(l1_rmse, label='Train')
plt.plot(l1_test_rmse, label='Test')
plt.title('L1 Regularization')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.savefig('./plots/q1_regl1.png', bbox_inches='tight', pad_inches=0.5)
plt.close()

print('RMSE on Train Set: %s' % (l1_rmse[-1]))
print('RMSE on Test Set: %s' % (l1_test_rmse[-1]))


# In[20]:


print('Linear Regression: ', val_rmse_collection[test_fold, -1])
print('L2 Regression: ', l2_test_rmse[-1])
print('L1 Regression: ', l1_test_rmse[-1])

