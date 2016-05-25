#!/usr/bin/pyhton
from ROOT import gROOT, TCanvas, TF1, TGraphErrors, TVirtualFitter, TMatrixD
import ROOT
import numpy as np

def func(x):
  return [0]+[1]*x

f = TF1("func", "[0]*x+[1]",10,20)

npoints = 10
xof = 10

datapoints = np.random.uniform(xof, xof + int(npoints) - 1, npoints)

mu = 0.0
sigma = 0.5
N = len(datapoints)
dataerrors = np.random.normal(mu, sigma, N)
y = datapoints + dataerrors
error_x = np.array([0.1 for _ in datapoints])
error_y = np.array([0.5 for _ in datapoints])
canvas = TCanvas()
print type(y)
print type(datapoints)
print type(error_x)
print type(error_y)

Graph_Errors = TGraphErrors(npoints, datapoints, y, error_x, error_y)
Graph_Errors.Fit(f,"V")
Graph_Errors.Draw('AP')



#Get the covariance and correlation matrices
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

print ("-----------------------")
print f.GetParameter(0)
print f.GetParameter(1)
print f.GetParError(0)
print f.GetParError(1)
print ("-----------------------")

fehler1_cormat[1][1]


f_error_plus = TF1("func", "[0]*x + [1] + sqrt([2]*[2] + ([3]*x)**2)",10,20)
f_error_minus = TF1("func_error_minus", "[0]*x + [1] - sqrt([2]*[2] + ([3]*x)**2)",10,20)

f_error_plus.SetParameter(0,f.GetParameter(0))
f_error_plus.SetParameter(1,f.GetParameter(1))
f_error_plus.SetParameter(2,f.GetParError(0))
f_error_plus.SetParameter(3,f.GetParError(1))

f_error_minus.SetParameter(0,f.GetParameter(0))
f_error_minus.SetParameter(1,f.GetParameter(1))
f_error_minus.SetParameter(2,f.GetParError(0))
f_error_minus.SetParameter(3,f.GetParError(1))

f_error_plus.Draw("same")
f_error_minus.Draw("same")


#min_delta_y = TF1("min_delta_y", make_band(f,datapoints), 10, 20)
#min_delta_y.Draw('AP')


canvas.Print("fit.png")