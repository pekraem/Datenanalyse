#Claculate Chi2
def chi2(a,b):
  chi=0.
  for i in a:
    chi += ((i-b)**2)/b
  return chi

#Input Data
tracks = range(1,9)
winners = [29, 19, 18, 25, 17, 10, 15, 11]

#print tracks, len(tracks)
#print winners, len(winners)

#Calculate the mean of the winners per track
mean = 0.
for n in winners:
  mean += n
mean /= len(winners)
#Significance
sig = [14.07, 18.48]
  
print mean

print sig

print chi2(winners,mean)

if sig[1]>=chi2(winners,mean):
  print "Significane level 1%: The hypothesis is true"
else:
  print "Significane level 1%: The hypothesis is not true"

if sig[0]>=chi2(winners,mean):
  print "Significane level 5%: The hypothesis is true"
else:
  print "Significane level 5%: The hypothesis is not true"
