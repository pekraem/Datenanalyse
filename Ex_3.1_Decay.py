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

c3 = ROOT.TCanvas("c3","c3",800,600)
h3 = ROOT.TH1F("h3", "h3", n_bin, 0, 2)
h4 = ROOT.TH1F("h4", "h4", n_bin, 0, 2)
h5 = ROOT.TH1F("h5", "h5", n_bin, 0, 2)

lam=1.   
num=[5,10,100]
for i in range(1000):
 for n in num:
  l = 1.
  for k in range(n):
   tmp = r.Rndm()
   l = l + log(lam*tau)
  if n==num[0]: 
   h3.Fill(l/n)
  elif n==num[1]:
   h4.Fill(l/n)
  elif n==num[2]:
   h5.Fill(l/n)

h3.Draw('hist')
h4.SetLineColor(ROOT.kBlue)
h4.Draw('same hist')
h5.SetLineColor(ROOT.kGreen)
h5.Draw('same hist')

close = raw_input()
