from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double, TFitResult, TColor
import numpy as np
import ROOT

nPoints = 20
data_x = np.array([ -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float)
data_y = np.array([5.0935, 2.1777, 0.2089, -2.3949, -2.4457, -3.0430, -2.2731, -2.0706, -1.6231, -2.5605, -0.7703, -0.3055, 1.6817, 1.8728, 3.6586, 3.2353, 4.2520, 5.2550, 3.8766, 4.2890 ], dtype=np.float)
ey=np.array(len(data_x)*[0.5],dtype=np.float)
ex=np.array(len(data_y)*[0],dtype=np.float)
xmin = min(np.min(d) for d in data_x)
xmax = max(np.max(d) for d in data_x)
fitfuncs=[]
legfuncs=[]

# define polynomials
P_2 = "[0] + [1]*x + [2]*x**2"
P_3 = "[0] + [1]*x + [2]*x**2 + [3]*x**3"
P_4 = "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4"
P_5 = "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5"
P_6 = "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5 + [6]*x**6"
P_7 = "[0] + [1]*x + [2]*x**2 + [3]*x**3 + [4]*x**4 + [5]*x**5 + [6]*x**6 + [7]*x**7"
polynom = [P_2,P_3,P_4,P_5,P_6,P_7]

# define Legendre polynomials
L_2 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.)"
L_3 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x)"
L_4 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.)"
L_5 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x)"
L_6 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x) + [6]*0.0625*(231.*x**6 - 315.*x**4 + 105.*x**2 - 5.)"
L_7 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]*0.5*(5.*x**3 - 3.*x) + [4]*0.125*(35.*x**4 - 30.*x**2 + 3.) + [5]*0.125*(63.*x**5 - 70.*x**3 + 15.*x) + [6]*0.0625*(231.*x**6 - 315.*x**4 + 105.*x**2 - 5.) + [7]*0.0625*(429.*x**7 - 693.*x**5 + 315.*x**3 -35.*x)"
legendre = [L_2,L_3,L_4,L_5,L_6,L_7]
# --------------------------------------------------------------------


c = TCanvas('c','c',800,600) 
gr = TGraphErrors(nPoints, data_x, data_y, ex, ey);
gr.Draw("AP")
#gStyle->SetPalette(55)

for i in range(len(polynom)):
#for i in range(1):
  #colnum = 0
  #col = "ROOT.kRed+"+str(colnum)
  exec('f_'+str(i)+' = TF1("f'+str(i)+'", polynom[i], xmin, xmax)')
  fitfuncs.append(eval('f_'+str(i)))
  #f = "f"+str(i)
  gr.Fit("f"+str(i),"V")
  #gr.GetFunction("f").SetLineColor(kRed+colnum)
  gr.Draw("APSAME")
  c.Update()
  #colnum += 1
  
print fitfuncs  
c.Update()
for func in fitfuncs:
  func.Draw("SAME")
c.Update()
raw_input()


c2 = TCanvas('c2','c2',800,600) 
gr2 = TGraphErrors(nPoints, data_x, data_y, ex, ey);
gr2.Draw("AP")
#gStyle->SetPalette(55)

for i in range(len(legendre)):
#for i in range(1):
  exec('l_'+str(i)+' = TF1("l'+str(i)+'", legendre[i], xmin, xmax)')
  legfuncs.append(eval('f_'+str(i)))
  gr2.Fit("l"+str(i),"V")
  gr2.Draw("APSAME")
  c2.Update()
  
print legfuncs  
c2.Update()
for func in legfuncs:
  func.Draw("SAME")
c2.Update()
raw_input()