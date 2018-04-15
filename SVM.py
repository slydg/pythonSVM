import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import exp

"""
# define the functions needed
"""
# build training set and testig set
def dataset(X):
    train = X.sample(int(len(X) / 3))
    test = X.drop(train.index.values)
    return train, test

# define the kernel function
def kernel(X1, X2, k=1.0):
    return exp(np.linalg.norm(X1 - X2) ** 2 / (-2 * k ** 2))
    # return X1.dot(X2)

# define support vector machine training function
# X is the training data and Y is the label of taining data
# max_passess is the max of times to iterate over alpha`s without changing
# C is the regularization parameter
# tol is the numerical tolerance 
def SVM(X, Y, C, max_passes, tol, use_kernel=False):
    m = len(X)
    alpha = np.zeros(m)
    b = 0
    passes = 0
    #Use SMO to find w and b
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            if not use_kernel:
                Ei = np.sum(alpha * Y * X.dot(X[i])) + b - Y[i]
            else:
                Ei = 0
                for k in range(len(X)):
                    Ei += alpha[k] * Y[k] * kernel(X[k],X[i])
                Ei += b - Y[i]
            if (Y[i] * Ei < -tol and alpha[i] < C ) or (Y[i] * Ei > tol and alpha[i] > 0 ):
                j = 0
                while True:
                    j = random.randrange(0,m,1)
                    if j != i:
                        break
                if not use_kernel:
                    Ej = np.sum(alpha * Y * X.dot(X[j])) + b - Y[j]
                else:
                    Ej = 0
                    for k in range(len(X)):
                        Ej += alpha[k] * Y[k] * kernel(X[k], X[j])
                    Ej += b - Y[j]
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                L = 0
                H = 0
                if Y[i] != Y[j]:
                    L = max(0,(alpha[j]-alpha[i]))
                    H = min(C,(C+alpha[j]-alpha[i]))
                else:
                    L = max(0,(alpha[i]+alpha[j]-C))
                    H = min(C,alpha[i]+alpha[j])
                if L == H:
                    continue
                if not use_kernel: 
                    mu = 2 * (X[i].dot(X[j])) - (X[i].dot(X[i])) - (X[j].dot(X[j]))
                else:
                    mu = 2 * kernel(X[i], X[j]) - \
                        kernel(X[i], X[i]) - kernel(X[j], X[j])
                if mu >= 0:
                    continue
                alpha_j = alpha[j] - (Y[j] * (Ei - Ej)) / mu
                if alpha_j > H:
                    alpha[j] = H
                elif L <= alpha_j <= H:
                    alpha[j] = alpha_j
                else:
                    alpha[j] = L
                if abs(alpha[j] - alpha_j_old) < 10 ** -5:
                    continue
                alpha[i] = alpha[i] + Y[i] * Y[j] * (alpha_j_old - alpha[j])
                if not use_kernel:
                    b1 = b - Ei - Y[i] * (alpha[i] - alpha_i_old) * X[i].dot(X[i]) - \
                        Y[j] * (alpha[j] - alpha_j_old) * X[i].dot(X[j])
                    b2 = b - Ej - Y[i] * (alpha[i] - alpha_i_old) * X[i].dot(X[j]) - \
                        Y[j] * (alpha[j] - alpha_j_old) * X[j].dot(X[j])
                else:
                    b1 = b - Ei - Y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[i]) - \
                        Y[j] * (alpha[j] - alpha_j_old) * kernel(X[i], X[j])
                    b2 = b - Ej - Y[i] * (alpha[i] - alpha_i_old) * kernel(X[i], X[j]) - \
                        Y[j] * (alpha[j] - alpha_j_old) * kernel(X[j], X[j])
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas = num_changed_alphas + 1
        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0
    if not use_kernel:
        w = (alpha * Y).dot(X)
        return w, b
    else:
        return  alpha, b

# define the hypothesis function (without kernel)
def h(X, w, b):
    return X.dot(w) + b

# define the hypothesis function (with kernel)
def h_with_kernel(alpha, b, X, training_X, training_Y):
    pre_Y = []
    for k in range(len(X)):
        y = 0
        for i in range(len(training_X)):
            y += alpha[i] * training_Y[i] * kernel(training_X[i], X[k])
        y += b
        pre_Y.append(y)
    return np.array(pre_Y)

# define the accruracy function 
def accr(pre_Y, te_Y):
    for i in range(len(pre_Y)):
        if pre_Y[i] <= 0:
            pre_Y[i] = -1
        else:
            pre_Y[i] = 1 
    y = pre_Y - te_Y
    return np.sum(y == 0) / len(y)

"""
build dataset
"""
X1 = np.random.randn(2, 30) + 1
X2 = np.random.randn(2, 30) - 1
Y1 = np.ones(30)
Y2 = -1 * np.ones(30)
x1 = pd.Series(np.array([X1[0], X2[0]]).reshape(60))
x2 = pd.Series(np.array([X1[1], X2[1]]).reshape(60))
y = pd.Series(np.array([Y1,Y2]).reshape(60))
data = pd.DataFrame(np.array([x1,x2,y]).T, columns=['x1','x2','label'])
training, testing = dataset(data)

tr_X = np.array(training[['x1','x2']])
tr_Y = np.array(training['label'])
te_X = np.array(testing[['x1', 'x2']])
te_Y = np.array(testing['label'])

"""
using SVM without kernel
"""
# train SVM to find the w and b
w, b = SVM(tr_X, tr_Y, 0.5, 200, 0.0001)
print(w)
print(b)

# the predicting label of the testing data
pre_Y = h(te_X, w, b)

#draw the testing data and the edge of two classes
red = testing[testing.label > 0]
blue = testing[testing.label < 0]
rx = red['x1'].values
ry = red['x2'].values
bx = blue['x1'].values
by = blue['x2'].values

plt.figure()
plt.plot(rx, ry, '.', color='red')
plt.plot(bx, by, '.', color='blue')

x = np.linspace(-3, 3, 5)
y = -((w[0] / w[1]) * x) - b/w[1]
plt.plot(x, y)

print("without kernel, the accuracy is ", accr(pre_Y, te_Y))

"""
using SVM with kernel
"""
# training SVM
alpha, b = SVM(tr_X, tr_Y, 0.5, 200, 0.0001, use_kernel=True)

# the predicting label of the testing data
pre_Y_with_kernel = h_with_kernel(alpha, b, te_X, tr_X, tr_Y)

print("with kernel, the accuracy is ", accr(pre_Y_with_kernel, te_Y))

"""
show the picture
"""
plt.show()
