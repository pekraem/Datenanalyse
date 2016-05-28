#!/usr/bin/env python
from ROOT import *
import numpy as np
from array import array as arr
import matplotlib.pyplot as plt
from numpy.linalg import inv


def drawCovEllipse(y,cov):
  ## draw coordinate system, line and center
  h = TH1F("h", "Covariance ellipse", 100, 6, 10.5)
  h.SetMinimum(6)
  h.SetMaximum(10.5)
  h.SetStats(False)
  h.Draw()
  #line = TLine(6, 6, 10.5, 10.5)
  #line.Draw()
  dot = TPolyMarker(1)
  dot.SetPoint(0, y[0], y[1])
  dot.SetMarkerStyle(kPlus)
  dot.Draw()
  nPoints = 200

  # calculate the ellipse
  W =cov
  print W
  ellipse1 = TGraph(nPoints+1)
  for iPhi in range(201):
    phi = iPhi * 2 * np.pi / 200.
    v = [0,0]
    v[0] = np.cos(phi)
    v[1] = np.sin(phi)
    sum = 0.
    for i in range(2):
      for j in range(2):
		sum += v[i] * W[i][j] * v[j]
    r = np.sqrt(1./sum)
    ellipse1.SetPoint(iPhi, y[0]+r*v[0], y[1]+r*v[1])
  ellipse1.Draw("L")
  gPad.Update()
  raw_input()


# --> this is the definition of the function to minimize, here a chi^2-function
def calcChi2mean(npar, mean):
    delta = np.array([[Double(m_w[0]) - mean[0]*mean[1]], [Double(m_w[1]) - mean[0]*mean[1]], [mean[1] - Double(m_w[2])]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matrix_inv, delta))
    return chi2

def calcChi2value(npar, mean):
    delta = np.array([[Double(m_w[0]*mean[1]) - mean[0]], [Double(m_w[1]*mean[1]) - mean[0]], [mean[1] - Double(m_w[2])]])
    delta_T = delta.T
    chi2 = np.dot(delta_T , np.dot(kor_matrix_inv, delta))
    return chi2
    

def fcn(npar, deriv, f, apar, iflag):
    f[0] = calcChi2mean(npar,apar) 


global m_w
m_w = [8.0, 8.5, 1.]
error = [0.02*8.0, 0.02*8.5, 0.1]

npar = 2 #maximum of npar parameters


global kor_matrix
kor_matrix=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(3):
  kor_matrix[i][i] = error[i]*error[i]

global kor_matrix_inv
kor_matrix_inv = inv(kor_matrix)


minuit = TMinuit(npar) #initialize TMinuit with a maximum of 2 params
minuit.SetFCN(fcn)
arglist = arr('d', 2*[0.01])
ierflg = Long(0)
arglist[0] = 1.

minuit.mnexcm("SET ERR", arglist, 1, ierflg)

# Set starting values and step sizes for parameters
vstart = [8., 1.]
step = [0.01 , 0.01]
minuit.mnparm(0, "y", vstart[0], step[0], 0,0,ierflg)
minuit.mnparm(1, "N", vstart[1], step[1], 0,0,ierflg)


# Now ready for minimization step
arglist[0] = 500
arglist[1] = 1.
minuit.mnexcm("MIGRAD", arglist ,2,ierflg)

drawCovEllipse(m_w,kor_matrix_inv)
