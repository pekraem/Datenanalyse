from ROOT import *
import numpy as np
from numpy.linalg import inv, eig

#b teil
f = [35, 218, 814, 1069, 651, 195, 18]
observed = [99, 386, 695, 877, 618, 254, 71]
R = np.array([[0.7,0.3,0.3,0.3,0.3,0.3,0.3],[0.3,0.4,0.3,0.3,0.3,0.3,0.3],[0.3,0.3,0.4,0.3,0.3,0.3,0.3],[0.3,0.3,0.3,0.4,0.3,0.3,0.3],[0.3,0.3,0.3,0.3,0.4,0.3,0.3],[0.3,0.3,0.3,0.3,0.3,0.4,0.3],[0.3,0.3,0.3,0.3,0.3,0.3,0.7]])
#for i in range(7):
    #for j in range(7):
        #if i == j:
            #R[i][j] = 0.4
        #else:
            #R[i][j] = 0.3
#R[0][0] = 0.7
#R[6][6] = 0.7

print R
R_inv = inv(R)
unfold = np.dot(R_inv, observed)
#------------------------------------------

print type(R_inv)



#c part
#digonalize R
lambda_value, U = eig(R)
print U[:,0]
print U[:,1]
print U[:,2]
print U[:,3]
print U[:,4]
print U[:,5]
print U[:,6]
for i in range(7):
  
print U
R_diag = np.dot(inv(U), np.dot(R, U)) #??? Warum nicht wie auf Blatt??

#construct g_diag and multi with 1/lambda_value
g_diag = np.dot(inv(U),observed)
lambda_inv = 1/lambda_value

for i in range(7):
  g_diag[i] = g_diag[i] * lambda_inv[i]
#g_diag = np.multiply(g_diag, 1/lambda_value)


# set lambda_reg and set elements of g_diag
lambda_reg = 0.2 #change it for c
for i in range(7):
  if lambda_value[i] < lambda_reg:
    g_diag[i] = 0

#calculate unfolded result
unfold = np.dot(U,g_diag)
















#file_1 = TFile("8b.root","RECREATE")
#g_histo = TH1F("g", "observed distribution", 7, 0, 7)
#f_histo = TH1F("f", "true distribution", 7, 0, 7)
#unfold_histo = TH1F("unfold", "unfolded distribution", 7, 0, 7)
#f_histo.SetFillColor(kRed)
#unfold_histo.SetFillColor(kBlue)


#for i in range(7):
    #for j in range(int(observed[i])):
        #g_histo.Fill(i)
    #for j in range(f[i]):
        #f_histo.Fill(i)
    #if range(int(unfold[i])) == []:
      #for j in range(int(unfold[i]*(-1))):
	#unfold_histo.Fill(i,-1)
    #else:
      #for j in range(int(unfold[i])):
	#unfold_histo.Fill(i)


#c2 = TCanvas("c2","Compare",900,700)
#c2.Divide(1,2)
#c2.cd(1)
#f_histo.Draw()

#g_histo.Draw("SAME")

#c2.cd(2)
#unfold_histo.Draw()
#file_1.Write()
#raw_input('Press <ret> to continue -> ')