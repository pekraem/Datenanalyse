import random as rndm

def chi2(a,b):
  chi=0
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

mean = 0
for n in winners:
  mean += n
mean /= len(winners)

sig = [14.07, 18.48]
  
print mean

print sig

print chi2(winners,mean)
