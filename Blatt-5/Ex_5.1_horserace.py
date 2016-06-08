import random as rndm
from scipy.stats import chisquare
from ROOT import TMath

def chi2(a,b):
  chi=0.
  for i in a:
    chi += (i-b)**2/b
  return chi


tracks = range(1,9)
winners = [29, 19, 18, 25, 17, 10, 15, 11]
#winners = []




print tracks, len(tracks)
print winners, len(winners)

#for i in range(8):
  #winners.append(rndm.gauss(18,10))

mean = 0.
for n in winners:
  mean += n
mean /= len(winners)


  
print mean

print winners

print chi2(winners,mean)

print TMath.Prob(chi2(winners,mean), 7)