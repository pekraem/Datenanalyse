from ROOT import *
import numpy as np
from numpy.linalg import inv

f = [35, 218, 814, 1069, 651, 195, 18]
observed = [99, 386, 695, 877, 618, 254, 71]
R = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
for i in range(7):
    for j in range(7):
        if i == j:
            R[i][j] = 0.4
        else:
            R[i][j] = 0.3
R[0][0] = 0.7
R[6][6] = 0.7
R_inv = inv(R)


unfold = np.dot(R_inv, observed)

file_1 = TFile("8b.root","RECREATE")
g_histo = TH1F("g", "observed distribution", 7, 0, 7)
f_histo = TH1F("f", "true distribution", 7, 0, 7)
unfold_histo = TH1F("unfold", "unfolded distribution", 7, 0, 7)
f_histo.SetFillColor(kRed)
unfold_histo.SetFillColor(kBlue)


for i in range(7):
    for j in range(int(observed[i])):
        g_histo.Fill(i)
    for j in range(f[i]):
        f_histo.Fill(i)
    if range(int(unfold[i])) == []:
      for j in range(int(unfold[i]*(-1))):
	unfold_histo.Fill(i,-1)
    else:
      for j in range(int(unfold[i])):
	unfold_histo.Fill(i)


c2 = TCanvas("c2","Compare",900,700)
c2.Divide(1,2)
c2.cd(1)
f_histo.Draw()

g_histo.Draw("SAME")

c2.cd(2)
unfold_histo.Draw()
file_1.Write()
raw_input('Press <ret> to continue -> ')
