from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double, gROOT
import numpy as np

reject = True
def fline(x,par):
  if (reject and x[0]<<1.3 and x[0]>>0.8):
    TF1.RejectPoint()
    return 0
  else:
    return par[0] + par[1]*x[0]


nPoints = 60
data_x = np.array(np.arange(0,3,0.05), dtype=np.float) # 3 GeV / 60 bins = 0.05 GeV per bin
data_y = np.array([6 ,1 ,10 ,12 ,6 ,13 ,23 ,22 ,15 ,21 ,23 ,26 ,36 ,25 ,27 ,35 ,40 ,44 ,66 ,81,  75 , 57 ,48 ,45 ,46 ,41 ,35 ,36 ,53 ,32 ,40 ,37 ,38 ,31 ,36 ,44 ,42 ,37 ,32 ,32, 43 ,44 ,35 ,33 ,33 ,39 ,29 ,41 ,32 ,44 ,26 ,39 ,29 ,35 ,32 ,21 ,21 ,15 ,25 ,15], dtype=np.float)
sigma_x = np.array(nPoints*[0], dtype=np.float)
sigma_y = np.array(np.sqrt(data_y), dtype=np.float)
xmin = min(np.min(d) for d in data_x)
xmax = max(np.max(d) for d in data_x)
pi = np.pi
print pi

L = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.) + [3]/pi*([4]/2)/((x-[5])**2+([4]/2)**2)"
L_2 = "[0] + [1]*x + [2]*0.5*(3.*x**2 - 1.)"

c = TCanvas('c','c',800,600) 
gr = TGraphErrors(nPoints, data_x, data_y, sigma_x, sigma_y);
gr.Draw("AP")

f = TF1('f',L,xmin,xmax)
f.SetParameter(3,90)
f.SetParameter(4,0.9)
f.SetParameter(5,0.1)
gr.Fit("f","V")
gr.Draw("APSAME")
c.Update()


maxval = f.GetMaximumX()
print maxval



gr.Fit('f','V')
reject = False
gr.Draw("APSAME")
c.Update()

fl = TF1("fl",fline,xmin,0.8,2)
fl.SetParameters(f.GetParameters())
gr.GetListOfFunctions().Add(fl)
gROOT.GetListOfFunctions().Remove(fl)


fr = TF1("fr",fline,1.3,xmax,2)
fr.SetParameters(f.GetParameters())
gr.GetListOfFunctions().Add(fr)
gROOT.GetListOfFunctions().Remove(fr)

gr.Draw("APSAME")

raw_input()
