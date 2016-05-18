import numpy as np
from ROOT import *

##______________________________________________________________________________
def testfit():

   gMinuit = TMinuit(1)
   gMinuit.SetFCN( fcn )

   arglist = array( 'd', 10*[0.] )
   ierflg = 1982

   arglist[0] = 1
   gMinuit.mnexcm( "SET ERR", arglist, 1, ierflg )

 # Set starting values and step sizes for parameters
   vstart = array( 'd', ( 3,  1,  0.1,  0.01  ) )
   step   = array( 'd', ( 0.1, 0.1, 0.01, 0.001 ) )
   gMinuit.mnparm( 0, "a1", vstart[0], step[0], 0, 0, ierflg )
   gMinuit.mnparm( 1, "a2", vstart[1], step[1], 0, 0, ierflg )
   gMinuit.mnparm( 2, "a3", vstart[2], step[2], 0, 0, ierflg )
   gMinuit.mnparm( 3, "a4", vstart[3], step[3], 0, 0, ierflg )

 # Now ready for minimization step
   arglist[0] = 500
   arglist[1] = 1.
   gMinuit.mnexcm( "MIGRAD", arglist, 2, ierflg )

 # Print results
   amin, edm, errdef = 0.18, 0.19, 0.20
   nvpar, nparx, icstat = 1983, 1984, 1985
   gMinuit.mnstat( amin, edm, errdef, nvpar, nparx, icstat )
   gMinuit.mnprin( 3, amin )


##______________________________________________________________________________
def fcn( npar, gin, f, par, iflag ):
   global ncount
   nbins = 1

 # calculate chisquare
   chisq, delta = 0., 0.
   for i in range(nbins):
      delta  = (m_w[i]-par)/errorw[i]
      chisq += delta*delta

   f[0] = chisq
   ncount += 1

 
 

m_w = [80457, 80448]


error_w1 = [30,11,47,17,17]
error_w2 = [33,12,0,19,17]



errorw = [



g1 = 0.
g2 = 0.

for i in range(len(error_w1)):
  g1 += error_w1[i]**2
  g2 += error_w2[i]**2

s = error_w1[3]*error_w2[3]
t = error_w1[4]*error_w2[4]

kor_matirx = np.array([[g1, s+t], [g2, s+t]])

mean = (m_w1 + m_w2)/2.

delta = np.array([[m_w1 - mean], [m_w2 - mean]])
delta_T = delta.T
chi2 = delta_T * kor_matirx * delta


print delta
print kor_matirx
print delta_T

print chi2