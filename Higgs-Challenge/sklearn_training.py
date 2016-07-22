from skxgb import xgbLearner
#from skLearner import sklearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages
from itertools import permutations
import numpy as np
import datetime

SKL_time_01 = clock()

#define variables, commend out unuseful ones
variables=["DER_mass_MMC",
	   "DER_mass_transverse_met_lep",
	   "DER_mass_vis",
	   "DER_pt_h",
           "DER_deltaeta_jet_jet",
           "DER_mass_jet_jet",
           "DER_prodeta_jet_jet",
           "DER_deltar_tau_lep",
           "DER_pt_tot",
           "DER_sum_pt",
           "DER_pt_ratio_lep_tau",
           "DER_met_phi_centrality",
           "DER_lep_eta_centrality",
           "PRI_tau_pt",
           "PRI_tau_eta",
           "PRI_tau_phi",
           "PRI_lep_pt",
           "PRI_lep_eta",
           "PRI_lep_phi",
           "PRI_met",
           "PRI_met_phi",
           "PRI_met_sumet",
           "PRI_jet_num",
           "PRI_jet_leading_pt",
           "PRI_jet_leading_eta",
           "PRI_jet_leading_phi",
           "PRI_jet_subleading_pt",
           "PRI_jet_subleading_eta",
           "PRI_jet_subleading_phi",
           "PRI_jet_all_pt"
]
varsss=['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'PRI_jet_leading_eta', 'PRI_tau_pt', 'DER_mass_vis', 'DER_deltar_tau_lep', 'PRI_lep_eta', 'PRI_tau_eta', 'DER_mass_jet_jet', 'PRI_met']

#construct scikit-learn classifier class (xgbLearner), initialize variables and paths
skl=xgbLearner(varsss)
skl.SetSPath("../../atlas-higgs-challenge-2014-v2_part.root")
skl.SetBPath("../../atlas-higgs-challenge-2014-v2_part.root")
#skl.SetStestPath("../../atlas-higgs-challenge-2014-v2_part.root")
#skl.SetBtestPath("../../atlas-higgs-challenge-2014-v2_part.root")
skl.SetSTreename('signal')
skl.SetBTreename('background')

#set BDT options
skl.SetGradBoostOption('n_estimators', 500)#n_estimators-->number of trees, that are trained
skl.SetGradBoostOption('max_depth', 2)#max_depht-->depht of trees that are trained
skl.SetGradBoostOption('learning_rate', 0.02)#learning_rate or shrinkage, value how much former Trees are weighted

#create Plotfile, were all plots are saved as pdf
skl.SetPlotFile()

#convert ROOT-Trees to numpy-Arrays (all saved inside skl)
skl.Convert()

#check conversion
print len(skl.test_var), len(skl.test_weights), len(skl.test_ID)
print len(skl.test_var[0]), len(varsss)
print skl.test_var[0], varsss

#create pairs for testing options
optslst = [(1900,0.25),(1900,0.2)]
#for i in range(20):
    #for j in range(20):
        #optslst.append(((i+1)*100,(1+j)*0.01))
#print optslst

best_ams = [0,'']

for opts in optslst:
    print opts
    SKL_time_01 = clock()
    
    skl.SetGradBoostOption('n_estimators', opts[0])
    skl.SetGradBoostOption('learning_rate', opts[1])
    
    #fit scikit learn classifier and return as train. now all functions of scikit-learn classifiers can be used with train. train can be given as train to many function of xgbLearner class
    train = skl.Classify()

    #predict BDT output and compute ams
    
    print len(skl.test_ID), len(train.decision_function(skl.test_var)), len(skl.test_weights)
    for i in range(len(skl.test_ID)):
        print skl.test_ID[i],train.decision_function(skl.test_var[i]),skl.test_weights[i]
    ams = skl.find_best_ams(skl.test_ID,train.decision_function(skl.test_var),skl.test_weights)
    


#print variable importances
#varlst=[]
#for i in range(len(variables)):
    #varlst.append((train.feature_importances_[i], variables[i]))
#variable_importance = sorted(varlst, key=lambda lst:lst[0] )
#print variable_importance
#for pair in variable_importance:
    #print pair[0],pair[1]
#var=['[']
#for i in range(10):
    #var.append(str(variable_importance[29-i][1]))
#var.append(']')
#print var

    SKL_time_02 = clock()
    time_elapsed = SKL_time_02-SKL_time_01
    
    logfile = open("../../logfile.txt","a+")
    logfile.write('######'+datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")+'#####\n\n'+skl.ReturnOpts()+'\n\n\n Best AMS_value:__________'+str(ams)+'\n ###############################################\n\n\n\n\n')
    logfile.close()
    
    if ams[0]>best_ams[0]:
        best_ams[0]=ams[0]
        best_ams[1]=skl.ReturnOpts()

    print 'ellapsed time during training:___', time_elapsed
    
logfile = open("../../logfile.txt","a+")
logfile.write('+++ Best AMS = ' + str(best_ams[0]) + '+++\nwith: ' + str(best_ams[1]) +'\n\n\n\n\n\n\n\n')
logfile.close()