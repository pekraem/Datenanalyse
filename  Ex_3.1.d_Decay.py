#!/usr/bin/env python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit, minimize

def func(x, t):
    return np.exp(-t*x)/t

def trans2(t,tau):
    return -np.log(1-t)/tau

#Likelihood function
def sigmoid(params):
    k = params[0]
    sd = params[1]

    yPred = np.exp(-k*(x))/k
    # Calculate negative log likelihood
    LL = -np.sum( stats.norm.logpdf(y, loc=yPred, scale=sd ) )

    return(LL)

#Part d

N=1000
data7=[]

for i in range(0,N):
    data7.append(trans2(np.random.rand(),1))


n,m,patches=plt.hist(data7,1000,range=[0, 10],normed=1,facecolor='g',log=False,alpha=0.1)

x=np.zeros(1000)
y=np.zeros(1000)
for i in range(0,1000):
    y[i]=n[i]
    x[i]=m[i]
#Least square method
[popt], pcov = curve_fit(func, x, y)#Not actually using least square/Should be worse for small N
print popt
#MLE
initParams = [1,1]

results = minimize(sigmoid, initParams, method='Nelder-Mead')


estParms = results.x
yOut = yPred = (np.exp(-estParms[0]*(x))/estParms[0])
print estParms[0]

#Plot the results
plt.xlabel('mean of tau') # axis labels
plt.ylabel('count')
plt.title('Histogram 3.1 d N=100000') # title
#plt.clf()
plt.plot(x, yOut,'b-')
plt.plot(x,func(x, popt),'r-')
plt.show()