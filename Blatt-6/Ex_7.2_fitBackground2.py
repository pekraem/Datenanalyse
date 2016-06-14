from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double, gROOT, TMatrixD
import numpy as np

#Define data
nPoints = 60
data_x = np.array(np.arange(0,3,0.05), dtype=np.float) # 3 GeV / 60 bins = 0.05 GeV per bin
data_y = np.array([6 ,1 ,10 ,12 ,6 ,13 ,23 ,22 ,15 ,21 ,23 ,26 ,36 ,25 ,27 ,35 ,40 ,44 ,66 ,81,  75 , 57 ,48 ,45 ,46 ,41 ,35 ,36 ,53 ,32 ,40 ,37 ,38 ,31 ,36 ,44 ,42 ,37 ,32 ,32, 43 ,44 ,35 ,33 ,33 ,39 ,29 ,41 ,32 ,44 ,26 ,39 ,29 ,35 ,32 ,21 ,21 ,15 ,25 ,15], dtype=np.float)
sigma_x = np.array(nPoints*[0], dtype=np.float)
sigma_y = np.array(np.sqrt(data_y), dtype=np.float)

pi=np.pi
L=TF1("L","[0]*([1]/2)/(pi*((x-[2])**2+([1]/2)**2))",0.6,1.2)
Pol=TF1("Pol","[0]*x**2+[1]*x+[2]",0,3)

#Fiting data with 6 parameters-------------------------------------------------------

gStyle.SetOptFit(111)  # superimpose fit results

c1=TCanvas("c1","Daten",200,10,700,500) #make nice Canvas 
c1.SetGrid()

gr=TGraphErrors(nPoints,data_x,data_y,sigma_x,sigma_y)
gr.SetTitle("6 Parameter Fit")
gr.Draw("AP");


Tmp=TF1("Tmp","[0]*x**2+[1]*x+[2]+[3]*([4]/2)/(3.141*((x-[5])**2+([4]/2)**2))",0,3)
Tmp.SetParameter(3,90)
Tmp.SetParameter(4,0.9)
Tmp.SetParameter(5,0.1)
gr.Fit(Tmp)
#c1.Update()
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

#Extracting background -----------------------------------

new_data_x=[]
new_data_y=[]
new_nPoints=0
j=0
for i in data_x:
  if i<=0.6:
    new_data_x.append(i)
    new_data_y.append(data_y[j])
    new_nPoints +=1
  elif i>=1.2:
    new_data_x.append(i)
    new_data_y.append(data_y[j])
    new_nPoints+=1
  j+=1

new_data_x=np.array(new_data_x,dtype=np.float)
new_data_y=np.array(new_data_y,dtype=np.float)
new_sigma_x = np.array(new_nPoints*[0], dtype=np.float)
new_sigma_y = np.array(np.sqrt(new_data_y), dtype=np.float)


gStyle.SetOptFit(111)  # superimpose fit results

c2=TCanvas("c2","Daten",200,10,700,500) #make nice Canvas 
c2.SetGrid()

gr=TGraphErrors(new_nPoints,new_data_x,new_data_y,new_sigma_x,new_sigma_y)
gr.SetTitle("Background Fit")
gr.Draw("AP");
gr.Fit(Pol)
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
#c1.Update()
gr.Draw('AP')
#raw_input('Press <ret> to continue -> ')
a=Pol.GetParameter(0)
b=Pol.GetParameter(1)
c=Pol.GetParameter(2)

def background(x,a,b,c):
  return a*x**2+b*x+c

#Fitting peak without beakground ------------------
new_data_x1=[]
new_data_y1=[]
new_nPoints1=0
j=0
for i in data_x:
  if i>0.6 and i<1.2:
    new_data_x1.append(i)
    new_data_y1.append(data_y[j]-background(i,a,b,c))
    new_nPoints1 +=1
  j+=1

new_data_x1=np.array(new_data_x1,dtype=np.float)
new_data_y1=np.array(new_data_y1,dtype=np.float)
new_sigma_x1 = np.array(new_nPoints*[0], dtype=np.float)
new_sigma_y1 = np.array(np.sqrt(np.abs(new_data_y1)), dtype=np.float)
#print new_data_x1
#print new_data_y1

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

