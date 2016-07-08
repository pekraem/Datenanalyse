import ROOT
#import sys
#sys.path.insert(0, '../../')
#from plotutils import *


#signal_train=Sample('signal training',ROOT.kRed,'/nfs/dust/cms/user/pkraemer/DATENANA_neu/atlas-higgs-challenge-2014-v2_part.root','')
#background_train=Sample('background training',ROOT.kBlue,'/nfs/dust/cms/user/pkraemer/DATENANA_neu/atlas-higgs-challenge-2014-v2_part.root','')


variables = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]


ROOT.gROOT.SetBatch(True)

f=ROOT.TFile("../../atlas-higgs-challenge-2014-v2_part.root")
S=f.Get("signal")
B=f.Get("background")
Ls=S.GetListOfBranches()
Lb=B.GetListOfBranches()
for ls in Ls:
  print ls

#create canvas and open plotfile  
c=ROOT.TCanvas("c","c",800,600)
c.Print("../../varhistos.pdf[")
c.Close()

#print histos
for ls, lb in zip(Ls, Lb):
  if ls.GetName()==lb.GetName():
    c=ROOT.TCanvas("c","c",800,600)
    
    print ls.GetName(), lb.GetName()
    S.Draw(ls.GetName()+">>hs","","HIST")
    B.Draw(lb.GetName()+">>hb","","HIST SAME")
    
    hs = ROOT.gROOT.FindObject("hs")
    hb = ROOT.gROOT.FindObject("hb")
    hs.SetLineColor(ROOT.kRed)
    hb.SetLineColor(ROOT.kBlue)
    hs.SetTitle(ls.GetName())    
    hs.Draw("HISTO")
    hb.Draw("HISTOSAME")    
    c.Update()
    c.Print("../../varhistos.pdf[")
    c.Close()
    
c=ROOT.TCanvas("c","c",800,600)    
c.Print("../../varhistos.pdf]")
c.Close()