import random as rndm

def chi2(a,b):
  chi=0
  for i in a:
    chi += (i-b)**2/b
  return chi


tracks = range(1,9)
#winners = [290, 190, 180, 250, 170, 100, 150, 110]
winners = []




print tracks, len(tracks)
print winners, len(winners)

for i in range(8):
  winners.append(rndm.gauss(18,10))

mean = 0
for n in winners:
  mean += n
mean /= len(winners)


  
print mean

print winners

print chi2(winners,mean)