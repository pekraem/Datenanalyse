import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

#Define transformation function for our problem
def trans1(t,tau):
    return -tau*np.log(1-t)
def trans2(t,tau):
    return -np.log(1-t)/tau
#Get random numbers
N=1000
M=10
data1=[]
for i in range(0,N):
    tmp=0
    for j in xrange(0,M):
        tmp=tmp+trans1(np.random.rand(),1)
    data1.append(tmp/10.0)

n,x,patches=plt.hist(data1,100,facecolor='g',log=False,alpha=0.5)
plt.xlabel('mean of tau') # axis labels
plt.ylabel('count')
plt.title('Histogram 3.1 b') # title
#plt.plot(x,np.exp(-x),'r-')
plt.show()
print "Mean 3.1 b:"
print np.mean(data1)

data2=[]
for i in range(0,N):
    tmp=0
    for j in xrange(0,5):
        tmp=tmp+trans2(np.random.rand(),1)
    data2.append(tmp/5.0)

n,x,patches=plt.hist(data2,100,facecolor='g',log=False,alpha=0.5)
plt.xlabel('mean of tau') # axis labels
plt.ylabel('count')
plt.title('Histogram 3.1 c N=5') # title
#plt.plot(x,np.exp(-x),'r-')
plt.show()
print "Mean 3.1 c N=5:"
print np.mean(data2)

data3=[]
for i in range(0,N):
    tmp=0
    for j in xrange(0,M):
        tmp=tmp+trans2(np.random.rand(),1)
    data3.append(tmp/10.0)

n,x,patches=plt.hist(data3,100,facecolor='g',log=False,alpha=0.5)
plt.xlabel('mean of tau') # axis labels
plt.ylabel('count')
plt.title('Histogram 3.1 c N=10') # title
#plt.plot(x,np.exp(-x),'r-')
plt.show()
print "Mean 3.1 c N=10:"
print np.mean(data3)

data4=[]
for i in range(0,N):
    tmp=0
    for j in xrange(0,100):
        tmp=tmp+trans2(np.random.rand(),1)
    data4.append(tmp/100.0)


n,x,patches=plt.hist(data4,100,facecolor='g',log=False,alpha=0.5)
plt.xlabel('mean of tau') # axis labels
plt.ylabel('count')
plt.title('Histogram 3.1 c N=100') # title
#plt.plot(x,np.exp(-x),'r-')
plt.show()
print "Mean 3.1 c N=100:"
print np.mean(data4)
