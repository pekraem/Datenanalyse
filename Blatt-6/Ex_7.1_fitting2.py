# ------------------------------------
#!/usr/bin/env python

# -------------------------------------------------------
# Draw a graph with error bars and fit a function to it
#                            (adapted from TGraphFit.C)
#--------------------------------------------------------


from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double, TFitResult, TColor, TMatrixD
import numpy as np
from array import array
#Define Data
nPoints = 20
data_x = np.array([ -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float)
data_y = np.array([5.0935, 2.1777, 0.2089, -2.3949, -2.4457, -3.0430, -2.2731, -2.0706, -1.6231, -2.5605, -0.7703, -0.3055, 1.6817, 1.8728, 3.6586, 3.2353, 4.2520, 5.2550, 3.8766, 4.2890 ], dtype=np.float)
ey=np.array(len(data_x)*[0.5],dtype=np.float)
ex=np.array(len(data_y)*[0],dtype=np.float)
xmin = min(np.min(d) for d in data_x)
xmax = max(np.max(d) for d in data_x)

P_2 =TF1("P_2", "[0] + [1]*x + [2]*x**2",-1,1)
P_3 =TF1("P_3",  "[0] + [1]*x + [2]*x**2 + [3]*x**3",-1,1)
P_4 =TF1("P_4",  "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4",-1,1)
P_5 =TF1("P_5",  "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5",-1,1)
P_6 =TF1("P_6",  "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5 + [6]*x**6",-1,1)
P_7 =TF1("P_7",  "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5 + [6]*x**6 + [7]*x**7",-1,1)
polynom = [P_2,P_3,P_4,P_5,P_6,P_7]
L_2 = TF1("L_2", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.)",-1,1)
L_3 = TF1("L_3", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x)",-1,1)
L_4 = TF1("L_4", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.)",-1,1)
L_5 = TF1("L_5", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x)",-1,1)
L_6 = TF1("L_6", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x) + [6]*0.0625*(231.*x**6 - 315.*x**4 + 105.*x**2 - 5.)",-1,1)
L_7 = TF1("L_7", "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x) + [6]*0.0625*(231.*x**6 - 315.*x**4 + 105.*x**2 - 5.) + [7]*0.0625*(429.*x**7 - 693.*x**5 + 315.*x**3 -35.*x)",-1,1)
legendre = [L_2,L_3,L_4,L_5,L_6,L_7]

#set some global options
gStyle.SetOptFit(111)  # superimpose fit results

c1=TCanvas("c1","Daten",200,10,700,500) #make nice Canvas 
c1.SetGrid()

#define some data points ...
ax = array('f',( -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) )
ay = array('f', (5.0935, 2.1777, 0.2089, -2.3949, -2.4457, -3.0430, -2.2731, -2.0706, -1.6231, -2.5605, -0.7703, -0.3055, 1.6817, 1.8728, 3.6586, 3.2353, 4.2520, 5.2550, 3.8766, 4.2890 ) ) 
ey=np.array(len(data_x)*[0.5],dtype=np.float)
ex=np.array(len(data_y)*[0],dtype=np.float)
# ... and pass to TGraphErros object
gr=TGraphErrors(nPoints,data_x,data_y,ex,ey)
gr.SetTitle("TGraphErrors mit Fit")
gr.Draw("AP");

Pol=["pol1","pol2","pol3","pol4","pol5","pol6","pol7"]

# Polifit
for i in range(len(polynom)):
  gr.Fit(polynom[i],"V")
  c1.Update()
  gr.Draw('AP')
  fitrp = TVirtualFitter.GetFitter()
  nPar = fitrp.GetNumberTotalParameters()
  covmat = TMatrixD(nPar, nPar,fitrp.GetCovarianceMatrix())
  print('The Covariance Matrix is: ')
  covmat.Print()
  cormat = TMatrixD(covmat)
  for i in range(nPar):
    for j in range(nPar):
      cormat[i][j] = cormat[i][j] / (np.sqrt(covmat[i][i]) * np.sqrt(covmat[j][j]))
  print('The Correlation Matrix is: ')
  cormat.Print()
  raw_input('Press <ret> to continue -> ')

#Legendre Poly
for i in range(len(legendre)):
  gr.Fit(legendre[i],"V")
  
  c1.Update()
  gr.Draw('AP')
  fitrp = TVirtualFitter.GetFitter()
  nPar = fitrp.GetNumberTotalParameters()
  covmat = TMatrixD(nPar, nPar,fitrp.GetCovarianceMatrix())
  print('The Covariance Matrix is: ')
  covmat.Print()
  cormat = TMatrixD(covmat)
  for i in range(nPar):
    for j in range(nPar):
      cormat[i][j] = cormat[i][j] / (np.sqrt(covmat[i][i]) * np.sqrt(covmat[j][j]))
  print('The Correlation Matrix is: ')
  cormat.Print()
  raw_input('Press <ret> to continue -> ')
# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')