from ROOT import gRandom, TGraphErrors, TF1, TMath, TVirtualFitter,  TCanvas, gStyle, TPaveStats,  TGraph, Double
import numpy as np

nPoints = 60
data_x = np.array(np.arange(0,3,0.05), dtype=np.float) # 3 GeV / 60 bins = 0.05 GeV per bin
data_y = np.array([6 ,1 ,10 ,12 ,6 ,13 ,23 ,22 ,15 ,21 ,23 ,26 ,36 ,25 ,27 ,35 ,40 ,44 ,66 ,81,  75 , 57 ,48 ,45 ,46 ,41 ,35 ,36 ,53 ,32 ,40 ,37 ,38 ,31 ,36 ,44 ,42 ,37 ,32 ,32, 43 ,44 ,35 ,33 ,33 ,39 ,29 ,41 ,32 ,44 ,26 ,39 ,29 ,35 ,32 ,21 ,21 ,15 ,25 ,15], dtype=np.float)
sigma_x = np.array(nPoints*[0], dtype=np.float)
sigma_y = np.array(np.sqrt(data_y), dtype=np.float)

