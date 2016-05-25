from ROOT import *
from array import array
#from ROOT import TF1, TRandom3, TGraphErrors, TCanvas, TVirtualFitter, TMatrixD, TMatrixDSym, TGraph,TH1F,TAttFill,TMinuit,TMatrix,TH1,TLine,TPolyMarker,TMath,gROOT
from math import pi, sqrt, fabs, exp
import numpy as np
import os
from math import pi

gROOT.Reset()

y = [8.0, 8.5]
ye = [8.0*0.02, 0.8, 8.5*0.02, 0.85]


cor = TMatrixD(2,2)
cor[0][0] = ye[0]**2 + ye[1]**2
cor[1][1] = ye[2]**2 + ye[3]**2
cor[1][0] = ye[1]*ye[3]
cor[0][1] = ye[1]*ye[3]

cor.Print()
cor.Invert()
cor.Print()



def chisquare(m):
      chi2 = (y[0]-m)*cor[0][0]*(y[0]-m)+(y[1]-m)*cor[1][1]*(y[1]-m)+(y[0]-m)*cor[0][1]*(y[1]-m)+(y[1]-m)*cor[1][0]*(y[0]-m)
      return chi2
    
def fu( npar, gin, f, par, iflag ):
  f[0] = chisquare(par[0])   
  
gMin = TMinuit(1)
gMin.SetFCN(fu)
gMin.DefineParameter( 0, "m", 8., 1., 0, 0 )
TMinuit.Migrad(gMin)


#draw coordinate system, line and center
h = TH1F("h", "1 and 2 sigma ellipse", 100, 6, 10.5)
h.SetMinimum(6)
h.SetMaximum(10.5)
h.SetStats(False)
h.Draw()
line = TLine(6, 6, 10.5, 10.5)
line.Draw()
dot = TPolyMarker(1)
dot.SetPoint(0, y[0], y[1])
dot.SetMarkerStyle(kPlus)
dot.Draw()
nPoints = 200

ellipse1 = TGraph(nPoints+1)
ellipse2 = TGraph(nPoints+1)
for iPhi in range(nPoints):
    phi = iPhi * 2*pi / 200
    v=2*[0]
    v[0] = cos(phi)
    v[1] = sin(phi)
    s = 0
    for i in range(2):
      for j in range(2):
	s += v[i] * cor[i][j] * v[j]
    r = sqrt(1./s)
    ellipse1.SetPoint(iPhi, y[0]+r*v[0], y[1]+r*v[1])
    ellipse2.SetPoint(iPhi, y[0]+2*r*v[0], y[1]+2*r*v[1])
ellipse1.Draw("LSAME")
ellipse2.Draw("LSAME")
raw_input()