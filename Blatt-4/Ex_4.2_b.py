#!/usr/bin/env python
from ROOT import *
import numpy as np
from array import array as arr
import matplotlib.pyplot as plt
from numpy.linalg import inv


def drawCovEllipse(y,cov):
  ## draw coordinate system, line and center

  h = TH1F("h", "1 and 2 sigma ellipse", 100, 6, 10.5)
  h.SetMinimum(6)
  h.SetMaximum(10.5)
  h.SetStats(false)
  h.Draw()
  line = TLine(6, 6, 10.5, 10.5)
  line.Draw()
  dot = TPolyMarker(1)
  dot.SetPoint(0, y[0], y[1])
  print y[0]
  dot.SetMarkerStyle(kPlus)
  dot.Draw()
  nPoints = 200

  # calculate the ellipse
  W =cov
  ellipse1 = TGraph(nPoints+1)
  ellipse2 = TGraph(nPoints+1)
  for iPhi in range(201):
    phi = iPhi * 2 * np.pi / 200.
    v = [0,0]
    v[0] = np.cos(phi)
    v[1] = np.sin(phi)
    sum = 0.
    for i in range(2):
      for j in range(2):
	sum += v[i] * W[i][j] * v[j]
    r = sqrt(1./sum)
    ellipse1.SetPoint(iPhi, y[0]+r*v[0], y[1]+r*v[1])
    ellipse2.SetPoint(iPhi, y[0]+2*r*v[0], y[1]+2*r*v[1])
  ellipse1.Draw("L")
  ellipse2.Draw("L")
  gPad.Update()
  raw_input()


# --> this is the definition of the function to minimize, here a chi^2-function
def calcChi2mean(npar, mean):
    delta = np.array([[Double(m_w[0]) - mean[1]*mean[2]], [Double(m_w[1]) - mean[1]*mean[2]], [Double(m_w[2]) - mean[2]]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matrix_inv, delta))
    return chi2

  
def calcChi2value(npar, mean):
    delta = np.array([[Double(m_w[0]*mean[2]) - mean[1]], [Double(m_w[1]*mean[2]) - mean[1]], [Double(m_w[2]) - mean[2]]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matrix_inv, delta))
    return chi2

  
def fcn(npar, deriv, f, apar, iflag):
    """ meaning of parametrs:
         npar:   number of parameters
         deriv:  aray of derivatives df/dp_i (x), optional
         f:      value of function to be minimised (typically chi2 or negLogL) 
         apar:   the array of parameters
         iflag:  internal flag: 1 at first call, 3 at the last, 4 during minimisation
    """
    f[0] = calcChi2value(npar,apar) 


    
global m_w
m_w = [8.0, 8.5, 1.]
error = [0.02*8.0, 0.02*8.5, 0.1]

npar = 1 #maximum of npar parameters


global kor_matrix
kor_matrix=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(3):
  kor_matrix[i][i] = error[i]*error[i]
global kor_matrix_inv
kor_matrix_inv = inv(kor_matrix)
print error[1]*error[1]
print kor_matrix
## --> set up MINUIT

#myMinuit = TMinuit(npar)  # initialize TMinuit with maximum of npar parameters
#myMinuit.SetFCN(fcn)      # set function to minimize
#arglist = arr('d', 2*[0.01]) # set error definition 
#ierflg = Long(0)             
#arglist[0] = 1.              # 1 sigma is Delta chi^2 = 1
#myMinuit.mnexcm("SET ERR", arglist ,1,ierflg)


## --> Set starting values and step sizes for parameters
## Define the parameters for the fit
#myMinuit.mnparm(0, "mean", 400., 0.001, 0,0,ierflg)
#myMinuit.mnparm(1, "N", 1., 0.001, 0,0,ierflg)


#arglist[0] = 6000 # Number of calls to FCN before giving up.
#arglist[1] = 0.3  # Tolerance
#myMinuit.mnexcm("MIGRAD", arglist ,2,ierflg)  # execute the minimisation

## --> check TMinuit status 
#amin, edm, errdef = Double(0.), Double(0.), Double(0.)
#nvpar, nparx, icstat = Long(0), Long(0), Long(0)
#myMinuit.mnstat(amin,edm,errdef,nvpar,nparx,icstat)
## meaning of parameters:
##   amin: value of fcn at minimum (=chi^2)
##   edm:  estimated distance to mimimum
##   errdef: delta_fcn used to define 1 sigma errors
##   nvpar: number of variable parameters
##   nparx: total number of parameters
##   icstat: status of error matrix: 
##           3=accurate 
##           2=forced pos. def 
##           1= approximative 
##           0=not calculated
#myMinuit.mnprin(3,amin) # print-out by Minuit

y = 0
N = 0
yerr = 0
Nerr = 0

minuit = TMinuit(npar)
minuit.SetFCN(fcn)

minuit.DefineParameter(0, "y", 400., 0.001, 0., 0.)
minuit.DefineParameter(1, "N", 1., 0.001, 0., 0.)

minuit.Migrad()





#nwerte = [m_w[0]-amin,m_w[1]-amin]
#drawCovEllipse(m_w,kor_matrix_inv)


