import ROOT
from math import *

#def Likelyhood(x, theta):
#  likely = 1.
#  for xi in x:
#    likely = likely * xi * theta
#    print likely
#  return likely


n_bin = 100
N = 10000

r = ROOT.TRandom3(0)
c = ROOT.TCanvas("c","c",800,600)
h = ROOT.TH1F("h","h", n_bin, 0 , 10)

tau = 1.

for i in range(N):
  tmp = r.Rndm()
  x = log(tau/tmp)
  h.Fill(x)
  
x=[]

h.Draw('hist')

#for b in range(n_bin):
#  x.append(h.GetBinContent(b))
#print x
  
#print Likelyhood(x, 1.)


h2 = ROOT.TH1F("h2", "h2", n_bin, 0, 2)

for i in range(1000):
 t=1.
 for k in range(10):
  tmp = r.Rndm()
  t = t + log(tau/tmp)
 h2.Fill(t/10.)

c2 = ROOT.TCanvas("c2","c2",800,600)
h2.Draw('hist')

#c3 = ROOT.TCanvas("c3","c3",800,600)
h3 = ROOT.TH1F("h3", "h3", n_bin, 0, 2)
h4 = ROOT.TH1F("h4", "h4", n_bin, 0, 2)
h5 = ROOT.TH1F("h5", "h5", n_bin, 0, 2)

lam=1.   
num=[5,10,100]
for i in range(1000):
 for n in num:
  l = 0
  for k in range(n):
   tmp = r.Rndm()
   l = l - log(lam*tmp)
  if n==num[0]: 
   h3.Fill(l/n)
  elif n==num[1]:
   h4.Fill(l/n)
  elif n==num[2]:
   h5.Fill(l/n)
#   print h5.GetEntries()
   
   
c3 = ROOT.TCanvas("c3","c3",800,600)

h5.Draw('hist')
#print 'test'
h4.SetLineColor(ROOT.kBlue)
h4.Draw('same hist')
h3.SetLineColor(ROOT.kRed)
h3.Draw('same hist')


#raw_input()
hd1 = ROOT.TH1F("hd1","hd1", 1000, 0 , 10)
hd2 = ROOT.TH1F("hd2","hd2", 1000, 0 , 10)
hd3 = ROOT.TH1F("hd3","hd3", 1000, 0 , 10)

histos = []
histos.append(hd1)
histos.append(hd2)
histos.append(hd3)

n_events=[10,1000,100000]


f1 = ROOT.TF1("f1", "1/[0]*exp(-x/[0])", 0, 10)
f2 = ROOT.TF1("f2", "1/[0]*exp(-x/[0])", 0, 10)
f3 = ROOT.TF1("f3", "1/[0]*exp(-x/[0])", 0, 10)

fkts_l=[]
fkts_l.append(f1)
fkts_l.append(f2)
fkts_l.append(f2)

f4 = ROOT.TF1("f4", "1/[0]*exp(-x/[0])", 0, 10)
f5 = ROOT.TF1("f5", "1/[0]*exp(-x/[0])", 0, 10)
f6 = ROOT.TF1("f6", "1/[0]*exp(-x/[0])", 0, 10)

fkts_c=[]
fkts_c.append(f4)
fkts_c.append(f5)
fkts_c.append(f6)


for h, n in zip(histos, n_events):
  print n
  for i in range(n):
    tmp = r.Rndm()
    x = log(tau/tmp)
    h.Fill(x)
    norm=h.GetEntries()
    h.Scale(1000/n)

c.Close()
c2.Close()
c3.Close()
c = ROOT.TCanvas("c","c",800,600)
histos[0].Draw()
histos[0].Fit("f1","L","",0,10)
raw_input()
c.Update()
histos[0].Fit("f4","","SAME",0,10)
raw_input()
c.Update()
histos[1].Draw()
c.Update()
raw_input()
histos[1].Fit("f2","L","",0,10)
raw_input()
c.Update()
histos[1].Fit("f5","L","SAME",0,10)
raw_input()
c.Update()
histos[1].Draw()
c.Update()
raw_input()
histos[2].Draw()
c.Update()
raw_input()
c.Update()
histos[2].Fit("f3","L","",0,10)
raw_input()
c.Update()
histos[2].Fit("f6","L","SAME",0,10)
raw_input()
c.Update()


close = raw_input()
