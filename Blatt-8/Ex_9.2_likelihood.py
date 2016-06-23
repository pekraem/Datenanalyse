from ROOT import *


#log_likelihood function:
def p_likeli(x,par):
  fac=1
  for n in range(int(round(par[0]))):
    fac*=(n+1)
  return -2*(TMath.Log(((x[0]**par[0])*TMath.Exp(-x[0]))/fac))


#2*delta log_likelihood
def p_likeli_cl(x,par):
  fac=1
  for n in range(int(round(par[0]))):
    fac*=(n+1)
  return (-2*par[1]-(TMath.Log(((x[0]**par[0])*TMath.Exp(-x[0]))/fac)))

log_likeli = TF1('log_likeli',p_likeli,0,10,1)

c = TCanvas('c','c',800,600)

log_likeli.SetParameter(0,3)
log_likeli.Draw()


c.SaveAs('log_likelihood.pdf')
m=log_likeli.GetMinimum()
xm=log_likeli.GetX(m,0,10)

print m, xm

c.Close()
print 'minimum is at v_t = ', m
c1 = TCanvas('c1','c1',800,600)

cl_likeli = TF1('cl_likeli',p_likeli_cl,0,30,2)
eins = TLine(0,1,30,1)
eins.SetLineColor(4)
zwei = TLine(0,1.28,30,1.28)
zwei.SetLineColor(3)
cl_likeli.SetParameter(0,3)
cl_likeli.SetParameter(1,m)
cl_likeli.Draw()
eins.Draw("SAME")
zwei.Draw("SAME")

#c1.SaveAs('delta_log_likelihood.pdf')
minlimit_68 = cl_likeli.GetX(1.,0,5)
maxlimit_68 = cl_likeli.GetX(1.,5,15)
print minlimit_68, maxlimit_68

minlimit_90 = cl_likeli.GetX(1.28,0,5)
maxlimit_90 = cl_likeli.GetX(1.28,5,15)
print minlimit_90, maxlimit_90

raw_input()
c1.Close()
