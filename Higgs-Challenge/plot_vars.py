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
#c.Close()

histos_s=[]
histos_b=[]

#print histos
for ls, lb in zip(Ls, Lb):
  if ls.GetName()==lb.GetName():
    #c=ROOT.TCanvas("c","c",800,600)
    
    #print ls.GetName(), lb.GetName()
    S.Draw(ls.GetName()+">>hists()")#,"","HIST SAME")
    B.Draw(lb.GetName()+">>histb()")#,"","HIST SAME")
    
    minxs = ROOT.gROOT.FindObject("hists").GetXaxis().GetXmin()
    minxb = ROOT.gROOT.FindObject("histb").GetXaxis().GetXmin()
    minx = min(minxs,minxb)
    print 'minx = ',minx
    
    maxxs = ROOT.gROOT.FindObject("hists").GetXaxis().GetXmax()
    maxxb = ROOT.gROOT.FindObject("histb").GetXaxis().GetXmax()
    maxx = max(maxxs, maxxb)
    print 'maxx = ',maxx
    
    maxys = ROOT.gROOT.FindObject("hists").GetMaximum()
    maxyb = ROOT.gROOT.FindObject("histb").GetMaximum()
    maxy = max(maxys, maxyb)
    print 'maxy = ',maxy
    
    h = ROOT.TH1F("h","h",100,minx,maxx)
    h.GetYaxis().SetRange(0,int(maxy))
    h.Draw()

    hs = ROOT.gROOT.FindObject("hists")
    hb = ROOT.gROOT.FindObject("histb")
    histos_s.append(hs.Clone())
    histos_b.append(hb.Clone())
    histos_s[-1]
    hs.SetLineColor(ROOT.kRed)
    hb.SetLineColor(ROOT.kBlue)
    hs.SetTitle(ls.GetName())
    #hs.GetXaxis().SetRange(int(minx), int(maxx))
    #hb.GetXaxis().SetRange(int(minx), int(maxx))
    #hs.GetYaxis().SetRange(0,int(maxy))
    #hb.GetYaxis().SetRange(0,int(maxy))
    #c.Close()
    #c=ROOT.TCanvas("c","c",800,600)
    hs.Draw("SAME")
    hb.Draw("SAME")    
    c.Update()
    ROOT.gPad.Modified()
    c.Print("../../varhistos.pdf")
    #c.Close()
    del h

#c=ROOT.TCanvas("c","c",800,600)    
c.Print("../../varhistos.pdf]")
c.Close()


c=ROOT.TCanvas("c","c",800,600)
c.Print("../../variables.pdf[")

for s, b in zip(histos_s,histos_b):
  s.SetLineColor(ROOT.kRed)
  s.Draw()
  b.SetLineColor(ROOT.kBlue)
  b.Draw("SAME")
  c.Update()
  c.Print("../../variables.pdf")
c.Print("../../variables.pdf]")  
  
