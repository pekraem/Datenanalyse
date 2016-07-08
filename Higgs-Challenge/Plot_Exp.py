from plotutils import *
import sys
sys.path.insert(0, '../..')
from plotutils import *

TreeToUse = ["../../atlas-higgs-challenge-2014-v2_part.root"]

VarToUse = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]


f=ROOT.TFile("../../atlas-higgs-challenge-2014-v2_part.root")
S=f.Get("signal")
B=f.Get("background")
Ls=S.GetListOfBranches()
Lb=B.GetListOfBranches()

Xmin=[0,0,0,0,0,0,-5,-5,0,0,0,-2,-2,0,-50,-5,0,0,-5,0,-5,0,0,0,-10,-10,0,-10,-10,0]
Xmax=[50,300,400,600,5,400,5,5,300,700,10,2,2,200,10,5,300,5,5,100,5,100,20,300,10,10,200,10,10,300]

colorlist = [ROOT.kRed, ROOT.kBlue]
OutputName = '../../variables.pdf'

##print histos
#for ls, lb in zip(Ls, Lb):
  #if ls.GetName()==lb.GetName():    
    ##print ls.GetName(), lb.GetName()
    #S.Draw(ls.GetName()+">>hists()")#,"","HIST SAME")
    #B.Draw(lb.GetName()+">>histb()")#,"","HIST SAME")
    
    #minxs = ROOT.gROOT.FindObject("hists").GetXaxis().GetXmin()
    #minxb = ROOT.gROOT.FindObject("histb").GetXaxis().GetXmin()
    #minx = min(minxs,minxb)
    #print 'minx = ',minx
    
    #maxxs = ROOT.gROOT.FindObject("hists").GetXaxis().GetXmax()
    #maxxb = ROOT.gROOT.FindObject("histb").GetXaxis().GetXmax()
    #maxx = max(maxxs, maxxb)
    #print 'maxx = ',maxx
    
    #maxys = ROOT.gROOT.FindObject("hists").GetMaximum()
    #maxyb = ROOT.gROOT.FindObject("histb").GetMaximum()
    #maxy = max(maxys, maxyb)
    #print 'maxy = ',maxy
    
    #Xmin.append(minx)
    #Xmax.append(maxx)


samples =[]
samples1 = []

samples.append(Sample(str(2),colorlist[0],TreeToUse[0],""))
samples1.append(Sample(str(1),colorlist[0],TreeToUse[0],""))

for Var in range(len(VarToUse)):
    plots = []
    histo = []

    plots.append(Plot(ROOT.TH1F(VarToUse[Var],VarToUse[Var],100,Xmin[Var],Xmax[Var]),VarToUse[Var],'')) #weight could create a bug, see plotutils 

    listOfSignalLists=createHistoLists_fromTree(plots,samples,'signal')
    listOfBackgroundLists=createHistoLists_fromTree(plots,samples1,'background')
    #print listOfSignalLists[0][0], listOfBackgroundLists[0][0]
    histo.append(listOfSignalLists[0][0])
    histo.append(listOfBackgroundLists[0][0])
    for i in range(len(histo)):
        histo[i].SetLineColor(colorlist[i])

    Canvas=drawHistosOnCanvas(histo,True,False,False,'histo',True,0,2) #last bool is ratio plot
    
    if VarToUse[Var]==VarToUse[0]: # original code by andrej
       Canvas.Print(OutputName+'[','pdf')
       Canvas.Print(OutputName,'pdf')
    elif VarToUse[Var]==VarToUse[-1]:
       Canvas.Print(OutputName,'pdf')
       Canvas.Print(OutputName+']','pdf')
    else:
       Canvas.Print(OutputName,'pdf')

    del plots
    del histo
