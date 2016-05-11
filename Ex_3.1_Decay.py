import ROOT

def Likelyhood(x, theta):
  likely = 1.
  for xi in x:
    likely = likely * xi * theta
  return likely


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

for b in range(nbins):
  x.append(h.GetBinContent(b))
  
print Likelyhood(x, 1)