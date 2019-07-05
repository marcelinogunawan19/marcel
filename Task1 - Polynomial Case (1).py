# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:34:35 2019

@author: MGu
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.formula.api as sm

n = 41
sigma = 2
a = ([5, 5, -2, 15, 0, 0, 0, 0])
x = np.arange(1, 41)
xsc = (2/n)*(x-(n/2))
eps = sigma*np.random.normal(0, 1, 40)
y = (a[0] +
     a[1]*xsc +
     a[2]*xsc**2 +
     a[3]*xsc**3 +
     a[4]*xsc**4 +
     a[5]*xsc**5 +
     a[6]*xsc**6 +
     a[7]*xsc**7 +
     eps)

# %% Create training and testing variables
X_train, X_test, y_train, y_test = train_test_split(xsc, y, test_size=0.67,
                                                    random_state=30)
X_train, y_train = zip(*sorted(zip(X_train, y_train)))
X_train = np.float64(X_train)
y_train = np.float64(y_train)

# Plot output all of the data
plt.figure()
plt.scatter(xsc, y)
plt.title('all data')

# Plot output training/test data
plt.figure()
plt.scatter(X_train, y_train, color='r')
plt.scatter(X_test, y_test, color='b')
plt.title('training/test data')


# %% Linear regression
lm = linear_model.LinearRegression().fit(X_train[:, np.newaxis],
                                         y_train[:, np.newaxis])
lm.intercept_
lm.coef_
Yhat = lm.predict(X_train[:,
                  np.newaxis])

# Plot output the Linear Regression
plt.figure()
plt.scatter(X_train, y_train,  color='black')
plt.plot(X_train, Yhat, color='blue', linewidth=3)

# %% Polynomial regression to be fitted between degrees 1 to 9
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.figure(figsize=(12, 7))
RMSEtrain = []
RMSEtest = []
for i in range(len(degrees)):
    ax = plt.subplot(3, 3, i + 1)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.6, wspace=0.35)

    # Create a fit a polynomial
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([('polynomial_features', polynomial_features),
                         ('linear_regression', linear_regression)])
    pipeline.fit(X_train[:, np.newaxis], y_train)

    ypredict = pipeline.predict(X_train[:, np.newaxis])
    MSE_sklearn = mean_squared_error(y_train, ypredict)
    RMSE = np.sqrt(MSE_sklearn)
    RMSEtrain.append(RMSE)

    ypredict = pipeline.predict(X_test[:, np.newaxis])
    MSE_sklearn2 = mean_squared_error(y_test, ypredict)
    RMSE2 = np.sqrt(MSE_sklearn2)
    RMSEtest.append(RMSE2)

    X_train2 = np.linspace(-0.9, 0.95, 100)  # generate points used to plot
    Y_trainnew = pipeline.predict(X_train2[:, np.newaxis])

    # plot output polynomial regression
    plt.plot(X_train2, Y_trainnew)
    plt.plot(X_train, y_train, 'ro')
    plt.xlabel('X_train')
    plt.ylabel('y_train')
    plt.title('Poly {}'.format(degrees[i]))

# plot output residual errors on test and training set
plt.figure()
plt.plot(degrees, RMSEtest, label='test', color='dodgerblue')
plt.plot(degrees, RMSEtrain, label='training', color='lightcoral')
plt.xlabel('model order')
plt.ylabel('rmse')
plt.title('residual errors on test and training set')
plt.legend()

# %% Summary of the polynomial regession
PS_IN = pd.DataFrame(columns=['Y', 'X'])
PS_IN['Y'] = y_train
PS_IN['X'] = X_train

var = ['I(X**{})'.format(i) for i in range(1, 10)]
var2 = [var[0]]
for i in range(1, 9):
    var2.append(var[i] + '+' + var2[i-1])
    test = formula = 'Y ~ ' + var2[i]
    result = sm.OLS.from_formula(test, data=PS_IN)
    results = result.fit()
    print(results.summary())


# %% Stepwise
# stepwise 1
sw1 = PolynomialFeatures(degree=degrees[1],
                         include_bias=False)
linear_regression = LinearRegression()
lm_sw1 = Pipeline([('sw1', sw1),
                   ('linear_regression', linear_regression)])
lm_sw1.fit(X_train[:, np.newaxis], y_train)

ypredicttrainsw1 = lm_sw1.predict(X_train[:, np.newaxis])
MSE_trainsw1 = mean_squared_error(y_train, ypredicttrainsw1)
RMSEtrainsw1 = np.sqrt(MSE_trainsw1)

ypredicttestsw1 = lm_sw1.predict(X_test[:, np.newaxis])
MSE_testsw1 = mean_squared_error(y_test, ypredicttestsw1)
RMSEtestsw1 = np.sqrt(MSE_testsw1)

# stepwise 2
sw2 = PolynomialFeatures(degree=degrees[3],
                         include_bias=False)
linear_regression = LinearRegression()
lm_sw2 = Pipeline([('sw2', sw2),
                   ('linear_regression', linear_regression)])
lm_sw2.fit(X_train[:, np.newaxis], y_train)

ypredicttrainsw2 = lm_sw2.predict(X_train[:, np.newaxis])
MSE_trainsw2 = mean_squared_error(y_train, ypredicttrainsw2)
RMSEtrainsw2 = np.sqrt(MSE_trainsw2)

ypredicttestsw2 = lm_sw2.predict(X_test[:, np.newaxis])
MSE_testsw2 = mean_squared_error(y_test, ypredicttestsw2)
RMSEtestsw2 = np.sqrt(MSE_testsw2)

# stepwise 3
sw3 = PolynomialFeatures(degree=degrees[7],
                         include_bias=False)
linear_regression = LinearRegression()
lm_sw3 = Pipeline([('sw3', sw3),
                   ('linear_regression', linear_regression)])
lm_sw3.fit(X_train[:, np.newaxis], y_train)

ypredicttrainsw3 = lm_sw3.predict(X_train[:, np.newaxis])
MSE_trainsw3 = mean_squared_error(y_train, ypredicttrainsw3)
RMSEtrainsw3 = np.sqrt(MSE_trainsw3)

ypredicttestsw3 = lm_sw3.predict(X_test[:, np.newaxis])
MSE_testsw3 = mean_squared_error(y_test, ypredicttestsw3)
RMSEtestsw3 = np.sqrt(MSE_testsw3)

# %% Ridge regression
Xtrain = []  # Xtrain is the 13x9 matrix
for i in range(1, 10):
        xtrain = X_train**i
        Xtrain.append(xtrain)

Xtrain = np.asarray(Xtrain)
Xtrain = np.transpose(Xtrain)

Xtest = []  # Xtest is the 27x9 matrix
for i in range(1, 10):
        xtest = X_test**i
        Xtest.append(xtest)

Xtest = np.asarray(Xtest)
Xtest = np.transpose(Xtest)

# Compute paths
n_alphas = 300
alphas = np.logspace(-5, 0, n_alphas)
coefs = []
errors = []
errorstest = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(Xtrain, y_train)
    coefs.append(ridge.coef_)
    Yhat = ridge.predict(Xtrain)
    Yhattest = ridge.predict(Xtest)
    errors.append(mean_squared_error(y_train, Yhat))
    errorstest.append(mean_squared_error(y_test, Yhattest))

# Ridge trace plot
plt.figure()
ax = plt.gca()
legends = ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.title('Ridge Trace')
plt.legend(iter(legends), ('X^1', 'X^2', 'X^3', 'X^4',
                           'X^5', 'X^6', 'X^7', 'X^8', 'X^9'))

# error vs ridge alphas
plt.figure()
ax = plt.gca()
ax.plot(alphas, errorstest, label='test', color='dodgerblue')
ax.plot(alphas, errors, label='training', color='lightcoral')
ax.set_xscale('log')
plt.xlabel('ridge alphas')
plt.ylabel('error')
plt.title('error vs ridge alphas')
plt.legend()
