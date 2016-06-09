from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double, gROOT
import numpy as np

reject = True
def fline(x,par):
  if (reject and x[0]<<1.3 and x[0]>>0.6):
    TF1.RejectPoint()
    return 0
  else:
    return par[0]+par[1]*x[0]+par[2]*0.5*(3.*x[0]**2 - 1.)

def func(x,par):
  return par[0]+par[1]*x[0]+par[2]*0.5*(3.*x[0]**2 - 1.)+par[3]/pi*(par[4]/2)/((x[0]-par[5])**2+(par[4]/2)**2)

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

f2 = TF1('f2',L,xmin,xmax)
f2.SetParameter(3,90)
f2.SetParameter(4,0.9)
f2.SetParameter(5,0.1)

maxval = f.GetMaximumX()
print maxval



gr.Fit('f2','V')
reject = False
gr.Draw("APSAME")
c.Update()

fl = TF1("fl",fline,xmin,0.8,3)
fl.SetParameters(f.GetParameters())
gr.GetListOfFunctions().Add(fl)
gROOT.GetListOfFunctions().Remove(fl)


fr = TF1("fr",fline,1.3,xmax,3)
fr.SetParameters(f.GetParameters())
gr.GetListOfFunctions().Add(fr)
gROOT.GetListOfFunctions().Remove(fr)

gr.Draw("APSAME")
c.Update()

#raw_input()

fnew = TF1('fnew', func ,xmin, xmax,6)
fnew.Add(fl,-1)
fnew.Add(fr,-1)
fnew.Draw()
c.Update()

raw_input()


def background(x,a,b,c):
  return a*x**2+b*x+c

#Fitting peak without beakground ----------------------------
new_data_x1=[]
new_data_y1=[]
new_nPoints1=0
#j=0
#for i in data_x:
  #if i>0.6 and i<1.3:
    #new_data_x1.append(i)
    #new_data_y1.append(data_y[j]-background(i,a,b,c))
    #new_nPoints1 +=1
  #j+=1

new_data_x1=np.array(new_data_x1,dtype=np.float)
new_data_y1=np.array(new_data_y1,dtype=np.float)
new_sigma_x1 = np.array(new_nPoints*[0], dtype=np.float)
new_sigma_y1 = np.array(np.sqrt(np.abs(new_data_y1)), dtype=np.float)
print new_data_x1
print new_data_y1

gStyle.SetOptFit(111)  # superimpose fit results

c3=TCanvas("c3","Daten",200,10,700,500) #make nice Canvas 
c3.SetGrid()

gr=TGraphErrors(new_nPoints1,new_data_x1,new_data_y1,new_sigma_x1,new_sigma_y1)
gr.SetTitle("Lorentzpeak Fit")
gr.Draw("AP");
L.SetParameter(0,9)
L.SetParameter(1,1)
L.SetParameter(2,1)
gr.Fit(L)
#c1.Update()
gr.Draw('AP')
raw_input('Press <ret> to continue -> ')