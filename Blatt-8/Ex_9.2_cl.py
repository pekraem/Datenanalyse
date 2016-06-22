from ROOT import *#TMath, TF1

# --------------------------------------------------------------------
# Returns the probability to observe n events if you expect s signal events
# and b background events
# Equivalent to: exp(-s-b)*(s+b)**n/TMath::Factorial(n)
def probability(n, s, b):
  return TMath.PoissonI(n,s+b)


# --------------------------------------------------------------------
# If you expect s signal events and b background events, return the
# probability to observe more than n events:
# used for classical upper-limits on the total number of events (signal + background): CL_SB
def pvalue_function_classical(x, par):
  s = x[0]
  n = par[0]
  b = par[1]
  
  pvalue_sum = 0.
  for i in range(int(round(n)+1)): # unfortunately n is passed as float and must be converted to int
    pvalue_sum += probability(i,s,b)
    
  if (s+b<0):
    pvalue_sum = 1.  # s+b should be positive
  
  return pvalue_sum


# --------------------------------------------------------------------
# If you expect s signal events and b background events, return the
# probability to observe more than n events:
# used for normalized upper-limits on the total number of signal events: CL_S = CL_SB / CL_B
def pvalue_function_cls(x, par):

  s = x[0]
  n = par[0]
  b = par[1]

  pvalue_sum = 0.
  for i in range(int(round(n)+1)): # unfortunately n is passed as float and must be converted to int
    pvalue_sum += probability(i,s,b)
    
  if (s+b<0):
    pvalue_sum = 1.  # s+b should be positive  
  
  pvalue_sum2 = 0.  #proability to observe n or less events if no signal is expected
  for i in range(int(round(n)+1)): # unfortunately n is passed as float and must be converted to int
    pvalue_sum2 += probability(i,0,b)
    
  result = pvalue_sum/pvalue_sum2
  
  return result



# --------------------------------------------------------------------
# Compute the value of s for which you reach the given p-value, if you
# observe n events and expect b background events
# set normalizeCLs to != 0 to use CL_S instead of CL_SB
def limit(n, b, pvalue, normalizeCLs = 0):
  
  #c = TCanvas('c','c',800,600)
  
  min = -1  # almost -infinity, since negetive values are not meaningful
  max = +10*(n+1)  # almost +infinity on relevant scale

  # function of one parameter varying between min and max
  pvf1 = TF1("pvf1",pvalue_function_classical,min,max,2)
  pvf2 = TF1("pvf2",pvalue_function_cls,min,max,2)

  pvf1.SetParameter(0,n)
  pvf1.SetParameter(1,b)
  pvf2.SetParameter(0,n)
  pvf2.SetParameter(1,b)
  
  #pvf1.Draw()
  #pvf2.Draw("same")
  #c.Update()
  #raw_input()
  limit = 0
  if normalizeCLs:
    limit = pvf2.GetX(pvalue,min,max)
    #pvf2.Draw("same")
    #c.Update()
  else:
    limit = pvf1.GetX(pvalue,min,max)
    #pvf1.Draw("same")
    #c.Update()
  return limit


def print_cl(x,par):
  b=x[0]
  n=par[0]
  pvalue=par[1]
  norm=par[2]
  return limit(n,b,pvalue,norm)
  
  

n_null=range(6)
c = TCanvas('c','c',800,600)
cl = TF1("CL",print_cl,0,6,3)

limits=[]

for n in n_null:
  cl.SetParameter(0,n)
  cl.SetParameter(1,0.9)
  cl.SetParameter(2,0)
  cl.SetLineColor(n+1)
  limits.append(cl.Clone())

limits[-1].Draw()
for tf in limits:
  tf.Draw("same")
c.SaveAs("Cl90p_00.pdf")
raw_input()  
c.Close()