from skxgb import xgbLearner
#from skLearner import sklearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages

SKL_time_01 = clock()

#define variables, commend out unuseful ones
variables=[#"DER_mass_MMC",
	   "DER_mass_transverse_met_lep",
	   "DER_mass_vis",
	   #"DER_pt_h",
           #"DER_deltaeta_jet_jet",
           #"DER_mass_jet_jet",
           #"DER_prodeta_jet_jet",
           "DER_deltar_tau_lep",
           #"DER_pt_tot",
           "DER_sum_pt",
           "DER_pt_ratio_lep_tau",
           "DER_met_phi_centrality",
           #"DER_lep_eta_centrality",
           "PRI_tau_pt",
           #"PRI_tau_eta",
           #"PRI_tau_phi",
           #"PRI_lep_pt",
           #"PRI_lep_eta",
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

#LEARNER=xgbLearner(variables)
XGB=xgbLearner(variables)
XGB.SetSPath("../../atlas-higgs-challenge-2014-v2.root")
XGB.SetBPath("../../atlas-higgs-challenge-2014-v2.root")
XGB.SetStestPath("../../atlas-higgs-challenge-2014-v2.root")
XGB.SetBtestPath("../../atlas-higgs-challenge-2014-v2.root")
XGB.SetSTreename('signal')
XGB.SetBTreename('background')
#XGB.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
#XGB.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
#XGB.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
#XGB.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
#XGB.SetSTreename("S")
#XGB.SetBTreename("B")
XGB.SetPlotFile()


#LEARN=sklearner(variables)
#LEARN.SetSPath("../../atlas-higgs-challenge-2014-v2.root")
#LEARN.SetBPath("../../atlas-higgs-challenge-2014-v2.root")
#LEARN.SetStestPath("../../atlas-higgs-challenge-2014-v2.root")#2D_test_scat.root')
#LEARN.SetBtestPath("../../atlas-higgs-challenge-2014-v2.root")#2D_test_scat.root')
#LEARN.SetSTreename('signal')
#LEARN.SetBTreename('background')
#LEARN.SetPlotFile()

names=[]
classifiers=[]
#LEARNER.Convert()

XGB.Convert()
#XGB.Shuffle(XGB.Var_Array,XGB.ID_Array)
#LEARN.Convert()
#opts=[['learning_rate',0.05,0.07],['n_estimators',1200,1500]]
#varlst, indexlist = LEARNER.permuteVars()
#print varlst
#print indexlist

#LEARNER.variateVars()

#LEARNER.SetGradBoostOption('n_estimators', 1500)
#LEARNER.SetGradBoostOption('max_depth', 2)
#LEARNER.SetGradBoostOption('learning_rate', 0.02)

XGB.SetGradBoostOption('n_estimators', 1500)
XGB.SetGradBoostOption('max_depth', 2)
XGB.SetGradBoostOption('learning_rate', 0.02)

#LEARN.SetGradBoostOption('n_estimators', 1200)
#LEARN.SetGradBoostOption('max_depth', 3)
#LEARN.SetGradBoostOption('learning_rate', 0.05)

t = XGB.Classify()#--> scikit-learn classification with predict_proba
#t2 = XGB.Classify()
#classifiers.append(t)
#names.append('GradientBoostingClassifier')
T = XGB.XGBClassify()#--> XGBoost Classification
#classifiers.append(T)
#names.append('XGBoostClassifier')
#LEARN.Output(t)
#LEARN.PrintOutput(t)
#LEARNER.permuteVars()
#XGB.testOpts(opts, 5)
#LEARN.PrintFigures()
#LEARNER.SetGradBoostDefault()
#XGB.PrintOutput(t)
#XGB.PrintOutput(t2)
#XGB.PrintOutput(T)

#XGB.ROCCurve(classifiers, names)
#XGB.CLFsCorrelation(classifiers, names)
#XGB.PrintOutput(t)
#XGB.PrintOutput(T)
#XGB.Output(t)
#XGB.Output(T)


#XGB.PrintOutput(t)
#XGB.Output(t)
#XGB.PrintFigures()
#T.evals_result()

#for var in XGB.test_var:
#	print T.predict(var)

#print LEARNER.KSTest(t)

#print XGB.KSTest(T)

#nsteps=10
#valuelist=[]
#for var in opts:
#  valuelist.append([])
#  name=var[0]
#  minv=var[1]
#  maxv=var[2]
#  currentvalue=minv
#  dstep=(maxv-minv)/nsteps
#  while currentvalue<=maxv:
#    valuelist[-1].append(currentvalue)
#    currentvalue+=dstep

#valuelist=[[0.1,...],[...]]

#combs



#compare decision_function and predict proba
#print t.decision_function(LEARN.test_var)
#print t.predict(LEARN.test_var)
#a = t.predict_proba(XGB.test_var)
#print a[:,1]
#b = a[:,1]
#print b[:(len(b)/2)]
#print b[(len(b)/2):]

#print t.predict_proba(XGB.test_var)[:,1]
#print t.decision_function(XGB.test_var)

sklpred = XGB.pred_proba(t)
ID=XGB.ID_Array
weights=XGB.Weights_Array
XGB.pred_proba(T)

XGB.find_best_ams(sklpred[0],ID,weights)