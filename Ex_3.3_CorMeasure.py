from ROOT import *
from array import array


mw = [80457., 80448.]
w1e = [30., 11., 47., 17., 17.]
w2e = [33., 12., 0., 19., 17.]
s1, s2 = 0., 0.

for i in range(len(w1e)):
  s1 += w1e[i]**2
  s2 += w2e[i]**2
sc = w1e[3]*w2e[4]+w1e[4]*w2e[3]

cor = TMatrixD(2,2)
cor[0][0] = s1
cor[1][1] = s2
cor[1][0] = sc
cor[0][1] = sc

cor.Invert()

cor.Print()



def fcn( npar, gin, f, par, iflag ):
  f[0] = chisquare(cor,mw,par)

def chisquare(cor,mw,par):
      chi2 = (mw[0]-par[0])*cor[0][0]*(mw[0]-par[0])+(mw[1]-par[0])*cor[1][1]*(mw[1]-par[0])+(mw[0]-par[0])*cor[0][1]*(mw[1]-par[0])+(mw[1]-par[0])*cor[1][0]*(mw[0]-par[0])
      return chi2
    
    

#arglist = array('d', 10*[0.])
#print arglist
#ierflg = array('i',[0])

gMin = TMinuit(1)
gMin.SetFCN(fcn)
#gMinuit.mnexcm( "SET ERR", arglist, 1, ierflg )
#mean = int((mw[0]+mw[1])/2)
#print mean

#gMinuit.mnparm( 0, "m_w", mean, 100, 79000, 81000, ierflg )
#arglist[0] = 1
#arglist[1] = 10
#gMinuit.mnexcm( "MIGRAD", arglist, 5, ierflg )
#gMinuit.Migrad()
gMin.DefineParameter( 0, "m_w", 80450, 100, 79000, 81000 )
gMin.Migrad()