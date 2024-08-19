import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 
import csv
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('automobil.csv')

print(df.describe())
cdf=df[['width','weight']]
print(cdf)

plt.scatter(df.width ,df.weight , color='blue')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()

msk = np.random.rand(len(df)) <0.8
train = cdf[msk]
test = cdf[~msk]


reg = linear_model.LinearRegression()

train_x = np.asanyarray(train[['width']])
train_y = np.asanyarray(train[['weight']])
reg.fit ( train_x , train_y)

print('coefs:', reg.coef_)
print('inter:' , reg.intercept_)

plt.scatter(train.width, train.weight , color='blue')
plt.plot( train_x , reg.coef_[0][0]*train_x + reg.intercept_[0] , '-r')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()



test_x = np.asanyarray(test[['width']])
test_y = np.asanyarray(test[['weight']])
test_y_= reg.predict(test_x)

print(" mean ab error: %.2f" % np.mean( np.absolute(test_y_ - test_y))) 
print(" residule sum of sq: %.2f(MSE)" % np.mean((test_y_ - test_y)**2)) 
print(" R2_score: %.2f" % r2_score(test_y , test_y_))



poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)


clf = linear_model.LinearRegression()
yhat = clf.fit ( train_x_poly , train_y)

print('coefs:', clf.coef_)
print('inter:' , clf.intercept_)

plt.scatter(train.width, train.weight , color='blue')
xx = np.arange(60.0, 73.0, 0.5)
plt.plot( xx,clf.coef_[0][2]*np.power(xx, 2) + clf.coef_[0][1]*xx +clf.intercept_[0] , '-r')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()

test_x_poly = poly.fit_transform(test_x)
poly_y = clf.predict(test_x_poly)

print("r2-score : %.2f" % r2_score(test_y, poly_y ))