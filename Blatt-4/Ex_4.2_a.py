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
def calcChi2(npar, mean):
    delta = np.array([[Double(m_w[0]) - mean[1]], [Double(m_w[1]) - mean[1]]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matirx_inv, delta)) - cqad
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

global cqad
cqad = 0
    
global m_w
m_w = [8.0, 8.5]
epsilon = 0.1
error = [0.02*8.0, 0.02*8.5]

npar = 1 #maximum of npar parameters

g1 = error[0]**2 + epsilon**2*m_w[0]**2
g2 = error[1]**2 + epsilon**2*m_w[1]**2

s = epsilon**2*m_w[0]*m_w[1]

global kor_matirx
kor_matirx = np.array([[g1, s], [s, g2]])
global kor_matirx_inv
kor_matirx_inv = inv(kor_matirx)

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



nwerte = [m_w[0]-amin,m_w[1]-amin]
drawCovEllipse(m_w,kor_matirx_inv)


