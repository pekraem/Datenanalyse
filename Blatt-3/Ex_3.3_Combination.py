#!/usr/bin/env python
from ROOT import TMinuit,Double,Long
import numpy as np
from array import array as arr
import matplotlib.pyplot as plt



# --> this is the definition of the function to minimize, here a chi^2-function
def calcChi2(npar, mean):
    delta = np.array([[Double(m_w[0]) - mean[1]], [Double(m_w[1]) - mean[1]]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matirx, delta))
    return chi2

def fcn(npar, deriv, f, apar, iflag):
    """ meaning of parametrs:
         npar:   number of parameters
         deriv:  aray of derivatives df/dp_i (x), optional
         f:      value of function to be minimised (typically chi2 or negLogL) 
         apar:   the array of parameters
         iflag:  internal flag: 1 at first call, 3 at the last, 4 during minimisation
    """
    f[0] = calcChi2(npar,apar) 
    
global m_w
m_w = [80457., 80448.]
error_w1 = [30.,11.,47.,17.,17.]
error_w2 = [33.,12.,0.,19.,17.]
npar = 1 #maximum of npar parameters

g1 = 0.
g2 = 0.
for i in range(len(error_w1)):
  g1 += error_w1[i]**2
  g2 += error_w2[i]**2

s = error_w1[3]*error_w2[3]
t = error_w1[4]*error_w2[4]

global kor_matirx
kor_matirx = np.array([[g1, s+t], [g2, s+t]])


# --> set up MINUIT
myMinuit = TMinuit(npar)  # initialize TMinuit with maximum of npar parameters
myMinuit.SetFCN(fcn)      # set function to minimize
arglist = arr('d', 2*[0.01]) # set error definition 
ierflg = Long(0)             
arglist[0] = 1.              # 1 sigma is Delta chi^2 = 1
myMinuit.mnexcm("SET ERR", arglist ,1,ierflg)


# --> Set starting values and step sizes for parameters
# Define the parameters for the fit
myMinuit.mnparm(1, "mean", 400, 0.001, 0,0,ierflg)

arglist[0] = 6000 # Number of calls to FCN before giving up.
arglist[1] = 0.3  # Tolerance
myMinuit.mnexcm("MIGRAD", arglist ,2,ierflg)  # execute the minimisation

# --> check TMinuit status 
amin, edm, errdef = Double(0.), Double(0.), Double(0.)
nvpar, nparx, icstat = Long(0), Long(0), Long(0)
myMinuit.mnstat(amin,edm,errdef,nvpar,nparx,icstat)
# meaning of parameters:
#   amin: value of fcn at minimum (=chi^2)
#   edm:  estimated distance to mimimum
#   errdef: delta_fcn used to define 1 sigma errors
#   nvpar: number of variable parameters
#   nparx: total number of parameters
#   icstat: status of error matrix: 
#           3=accurate 
#           2=forced pos. def 
#           1= approximative 
#           0=not calculated
myMinuit.mnprin(3,amin) # print-out by Minuit

