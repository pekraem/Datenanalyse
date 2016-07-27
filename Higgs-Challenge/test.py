from trainer import Trainer
import sys
sys.path.insert(1, '../pyroot-plotscripts')
from plotutils import *
from mvautils import *
#from interpol_vars import *

#define variables, commend out unuseful ones
variables=["DER_mass_MMC",
	   "DER_mass_transverse_met_lep",
	   "DER_mass_vis",
	   #"DER_pt_h",
           #"DER_deltaeta_jet_jet",
           "DER_mass_jet_jet",
           #"DER_prodeta_jet_jet",
           "DER_deltar_tau_lep",
           #"DER_pt_tot",
           "DER_sum_pt",
           "DER_pt_ratio_lep_tau",
           "DER_met_phi_centrality",
           #"DER_lep_eta_centrality",
           "PRI_tau_pt",
           "PRI_tau_eta",
           #"PRI_tau_phi",
           #"PRI_lep_pt",
           "PRI_lep_eta",
           #"PRI_lep_phi",
           "PRI_met",
           #"PRI_met_phi",
           #"PRI_met_sumet",
           #"PRI_jet_num",
           #"PRI_jet_leading_pt",
           "PRI_jet_leading_eta",
           #"PRI_jet_leading_phi",
           #"PRI_jet_subleading_pt",
           #"PRI_jet_subleading_eta",
           #"PRI_jet_subleading_phi",
           #"PRI_jet_all_pt"
]
varsss=['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'PRI_jet_leading_eta', 'PRI_tau_pt', 'DER_mass_vis', 'DER_deltar_tau_lep', 'PRI_lep_eta', 'PRI_tau_eta', 'DER_mass_jet_jet', 'PRI_met']

#needed for trainer class
addtional_variables=[]

#samples have a name, a color, a path, and a selection (not implemented yet for training)
#only the path is really relevant atm
cat='6j4t'

#change path if necessary
signal_test=Sample('t#bar{t}H test',ROOT.kBlue,"../../atlas-higgs-challenge-2014-v2.root",'') 
signal_train=Sample('t#bar{t}H training',ROOT.kGreen,"../../atlas-higgs-challenge-2014-v2_part.root",'')
background_test=Sample('t#bar{t} test',ROOT.kRed+1,"../../atlas-higgs-challenge-2014-v2.root",'')
background_train=Sample('t#bar{t} training',ROOT.kRed-1,"../../atlas-higgs-challenge-2014-v2_part.root",'')

#create trainer and set trainer options
trainer=Trainer(variables,addtional_variables)
trainer.addSamples(signal_train,background_train,signal_test,background_test) #add the sample defined above
trainer.setSTreeName('signal') # name of signaltree in files
trainer.setBTreeName('background') # name of backgroundtree in files
trainer.setReasonableDefaults() # set some configurations to reasonable values
trainer.setEqualNumEvents(True) # reweight events so that integral in training and testsample is the same
trainer.useTransformations(True) # faster this way
trainer.setVerbose(True) # no output during BDT training and testing
trainer.setWeightExpression('Weight')

#set BDT options
trainer.setBDTOption("NTrees=1200")
trainer.setBDTOption("Shrinkage=0.05")
trainer.setBDTOption("nCuts=50")
trainer.setBDTOption("MaxDepth=2")

print trainer.best_variables
trainer.trainBDT(variables)
ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()

print ROC, ROCT, ksS, ksB

trainer.bookReader()
print trainer.bdtoptions
print trainer.factoryoptions

#trainer.showGui()
