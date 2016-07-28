

import ROOT
import math
import sys
import os
import datetime
from array import array
from subprocess import call
sys.path.insert(0, '../pyroot-plotscripts')
from plotutils import *
from mvautils import *
from time import *
import numpy as np

class Trainer:
    def __init__(self, variables, variables_to_try=[], verbose=False):
        self.best_variables=variables
        self.variables_to_try=variables_to_try
        self.verbose=verbose
        self.ntrainings=0
        self.stopwatch=ROOT.TStopwatch()
        self.weightfile='weights/weights.xml'
        weightpath='../../'.join((self.weightfile.split('/'))[:-1])
        if not os.path.exists( weightpath ):
            os.makedirs(weightpath)
        self.outpath='../../outfile/'
        self.rootfile='autotrain.root'
        outfilepath='/'.join(((self.outpath+self.rootfile).split('/'))[:-1])
        if not os.path.exists( outfilepath ):
            os.makedirs(outfilepath)
        self.streename='MVATree'
        self.btreename='MVATree'
        self.weightexpression='1'
        self.equalnumevents=True
        self.selection=''
        self.factoryoptions="V:!Silent:Color:DrawProgressBar:AnalysisType=Classification:Transformations=I;D;P;G,D"
        self.bdtoptions= "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:!UseBaggedBoost:BaggedSampleFraction=0.6:nCuts=20:MaxDepth=2:NegWeightTreatment=IgnoreNegWeightsInTraining"     
        self.setVerbose(verbose)
        self.signal_prediction=[]
        self.background_prediction=[]
        self.treepath="../../atlas-higgs-challenge-2014-v2.root"

    def setVerbose(self,v=True):
        self.verbose=v
        if self.verbose:
            self.setFactoryOption('!Silent')
        else:
            self.setFactoryOption('Silent')
           
    def addSamples(self, signal_train,background_train,signal_test,background_test):
    #def addSamples(self, signal_train,background_train):      
        self.signal_train=signal_train
        self.signal_test=signal_test
        self.background_train=background_train
        self.background_test=background_test
        
    def setSelection(self, selection):
        self.selection=selection

    def setFactoryOption(self, option):
        self.factoryoptions=replaceOption(option,self.factoryoptions)

    def setBDTOption(self, option):
        self.bdtoptions=replaceOption(option,self.bdtoptions)

    def setEqualNumEvents(self, b=True):
        self.equalnumevents=b

    def setWeightExpression(self, exp):
        self.weightexpression=exp

    def setSTreeName(self, treename):
        self.streename=treename
        
    def setBTreeName(self, treename):
        self.btreename=treename

    def setReasonableDefaults(self):
        self.setBDTOption('MaxDepth=2')
        self.setBDTOption('nCuts=60')
        self.setBDTOption('Shrinkage=0.02')
        self.setBDTOption('NTrees=1000')
        self.setBDTOption('NegWeightTreatment=IgnoreNegWeightsInTraining')
        self.setBDTOption('UseBaggedBoost')
        self.equalnumevents=True

    def useTransformations(self, b=True):
        # transformation make the training slower
        if b:
            self.setFactoryOption('Transformations=I;D;P;G,D')
        else:
            self.setFactoryOption('Transformations=I')

    def showGui(self):
        ROOT.gROOT.SetMacroPath( "../../" )
        ROOT.gROOT.Macro       ( "./TMVAlogon.C" )    
        ROOT.gROOT.LoadMacro   ( "./TMVAGui.C" )

    def printVars(self):
        print self.best_variables


    # trains a without changing the defaults of the trainer
    def trainBDT(self,variables_=[],bdtoptions_="",factoryoptions_=""):
        if not hasattr(self, 'signal_train') or not hasattr(self, 'signal_test') or not hasattr(self, 'background_train')  or not hasattr(self, 'background_test'):
            print 'set training and test samples first'
            return
        fout = ROOT.TFile(self.outpath+self.rootfile,"RECREATE")
        # use given options and trainer defaults if an options is not specified
        newbdtoptions=replaceOptions(bdtoptions_,self.bdtoptions)
        newfactoryoptions=replaceOptions(factoryoptions_,self.factoryoptions)
        factory = ROOT.TMVA.Factory("TMVAClassification",fout,newfactoryoptions)
        # add variables
        variables=variables_
        if len(variables)==0:
            variables = self.best_variables
        for var in variables:
            factory.AddVariable(var)
        # add signal and background trees
        inputS = ROOT.TFile( self.signal_train.path )
        inputB = ROOT.TFile( self.background_train.path )          
        treeS = inputS.Get(self.streename)
        treeB = inputB.Get(self.btreename)
        
        #inputS_test = ROOT.TFile( self.signal_test.path )
        #inputB_test = ROOT.TFile( self.background_test.path )          
        #treeS_test = inputS_test.Get(self.streename)
        #treeB_test = inputB_test.Get(self.btreename)
        
        # use equal weights for signal and bkg
        signalWeight     = 1.
        backgroundWeight = 1.
        factory.AddSignalTree    ( treeS, signalWeight )
        factory.AddBackgroundTree( treeB, backgroundWeight)
        #factory.AddSignalTree    ( treeS_test, signalWeight,ROOT.TMVA.Types.kTesting )
        #factory.AddBackgroundTree( treeB_test, backgroundWeight,ROOT.TMVA.Types.kTesting)
        factory.SetWeightExpression(self.weightexpression)
        # make cuts
        mycuts = ROOT.TCut(self.selection)
        mycutb = ROOT.TCut(self.selection)
        # train and test all methods
        normmode="NormMode=NumEvents:"
        if self.equalnumevents:
            normmode="NormMode=EqualNumEvents:"
        factory.PrepareTrainingAndTestTree( mycuts, mycutb,
                                            "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:!V:"+normmode )
        #norm modes: NumEvents, EqualNumEvents
        factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDTG",newbdtoptions )
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        fout.Close()
        weightfile=self.weightfile
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        weightfile=weightfile.replace('.xml','_'+dt+'.xml')
        call(['cp','weights/TMVAClassification_BDTG.weights.xml',weightfile])
        movedfile=self.rootfile
        movedfile=movedfile.replace('.root','_'+dt+'.root')
        #call(['cp',self.rootfile,movedfile])
        self.trainedweight=weightfile

    def evaluateLastTraining(self):
        f = ROOT.TFile(self.outpath+self.rootfile)
    
        histoS = f.FindObjectAny('MVA_BDTG_S')
        histoB = f.FindObjectAny('MVA_BDTG_B')
        histoTrainS = f.FindObjectAny('MVA_BDTG_Train_S')
        histoTrainB = f.FindObjectAny('MVA_BDTG_Train_B')
        histo_rejBvsS = f.FindObjectAny('MVA_BDTG_rejBvsS')
        histo_effBvsS = f.FindObjectAny('MVA_BDTG_effBvsS')
        histo_effS = f.FindObjectAny('MVA_BDTG_effS')
        histo_effB = f.FindObjectAny('MVA_BDTG_effB')
        histo_trainingRejBvsS = f.FindObjectAny('MVA_BDTG_trainingRejBvsS')    

        rocintegral=histo_rejBvsS.Integral()/histo_rejBvsS.GetNbinsX()
        rocintegral_training=histo_trainingRejBvsS.Integral()/histo_trainingRejBvsS.GetNbinsX()
        bkgRej50=histo_rejBvsS.GetBinContent(histo_rejBvsS.FindBin(0.5))
        bkgRej50_training=histo_trainingRejBvsS.GetBinContent(histo_trainingRejBvsS.FindBin(0.5))
        ksS=histoTrainS.KolmogorovTest(histoS)
        ksB=histoTrainB.KolmogorovTest(histoB)
        
        ##implement better

        c1=ROOT.TCanvas("c1","c1",800,600)
        histoB.SetLineColor(ROOT.kBlue)
        histoTrainB.SetLineColor(ROOT.kRed)

        histoB.Draw("histo E")
        histoTrainB.Draw("SAME histo E")


        c1.SaveAs("Signal_07.pdf")
        
        outstr='ROC='+str(rocintegral)+'   ROC_tr='+str(rocintegral_training)+'   ksS='+str(ksS)+'   ksB'+str(ksB)+"\n"
        logfile = open("log_roc.txt","a+")
        logfile.write('######'+str(localtime())+'#####'+"\n"+"\n"+"\n"+'best AMS = ,'+str(self.bookbetterReader())+"\n"+str(self.best_variables)+"\n"+"\n"+str(self.bdtoptions)+"\n"+"\n"+outstr+'###############################################\n\n\n\n\n')
        logfile.close()


        return rocintegral, ksS, ksB, rocintegral_training
    
    def drawBDT(self):
        f = ROOT.TFile(self.rootfile)

        histoS = f.FindObjectAny('MVA_BDTG_S')
        histoB = f.FindObjectAny('MVA_BDTG_B')
        histoTrainS = f.FindObjectAny('MVA_BDTG_Train_S')
        histoTrainB = f.FindObjectAny('MVA_BDTG_Train_B')
        
        histoS.SetLineColor(self.signal_test.color)
        histoS.Draw('histo')
        histoB.SetLineColor(self.background_test.color)
        histoB.Draw('samehisto')
        histoTrainS.SetLineColor(self.signal_train.color)
        histoTrainS.Draw('same')
        histoTrainB.SetLineColor(self.background_train.color)
        histoTrainB.Draw('same')

    def removeWorstUntil(self,length):
        if(len(self.best_variables)<=length):
            return 
        else:
            print "####### findig variable to remove, nvars is "+str(len(self.best_variables))+", removing until nvars is "+str(length)+"."
            bestscore=-1.
            bestvars=[]
            worstvar=""
            for i in range(len(self.best_variables)):
                # sublist excluding variables i
                sublist=self.best_variables[:i]+self.best_variables[i+1:]
                print 'training BDT without',self.best_variables[i]
                self.trainBDT(sublist)
                score=self.evaluateLastTraining()
                print 'score',score
                if score>bestscore:
                    bestscore=score
                    bestvars=sublist
                    worstvar=self.best_variables[i]
            print "####### removing ",
            print worstvar
            self.variables_to_try.append(worstvar)
            self.best_variables=bestvars
            self.removeWorstUntil(length)

    def addBestUntil(self,length):
        if(len(self.best_variables)>=length):
            return
        elif len(self.variables_to_try)==0:
            return        
        else:
            print "####### findig variable to add, nvars is "+str(len(self.best_variables))+", adding until nvars is "+str(length)+"."
            bestscore=-1.
            bestvar=""
            for var in self.variables_to_try:
                newlist=self.best_variables+[var]
                print 'training BDT with',var
                self.trainBDT(newlist)
                score=self.evaluateLastTraining()
                print 'score:',score
                if score>bestscore:
                    bestscore=score
                    bestvar=var
            print "####### adding ",
            print bestvar
            self.variables_to_try.remove(bestvar)
            self.best_variables=self.best_variables+[bestvar]
            self.addBestUntil(length)
        

    def optimizeOption(self,option,factorlist=[0.3,0.5,0.7,1.,1.5,2.,3.]):
        currentvalue=float(getValueOf(option,self.bdtoptions))
        print "####### optimizing "+option+", starting value",currentvalue
        valuelist=[x * currentvalue for x in factorlist] 
        print "####### trying values ",
        print valuelist
        best=valuelist[0]
        bestscore=-1
        for n in valuelist:
            theoption=option+'='+str(n)
            print 'training BDT with',theoption
            self.trainBDT([],theoption)
            score=self.evaluateLastTraining()
            print 'score:',score
            if score>bestscore:
                bestscore=score
                best=n
        print "####### optiminal value is", best
        print "####### yielding scroe ", bestscore
        self.setBDTOption(option+'='+str(best))
        if best==valuelist[-1] and len(valuelist)>2:
            print "####### optiminal value is highest value, optimizing again"
            highfactorlist=[f for f in factorlist if f > factorlist[-2]/factorlist[-1]]
            self.optimizeOption(option,highfactorlist)
        if best==valuelist[0]and len(valuelist)>2:
            print "####### optiminal value is lowest value, optimizing again"
            lowfactorlist=[f for f in factorlist if f < factorlist[1]/factorlist[0]]            
            self.optimizeOption(option,lowfactorlist)


    def suche(self,NTrees_min, NTrees_max, Shrin_min, Shrin_max, Schritte):


	self.setBDTOption("NTrees="+str(NTrees_min))
	self.setBDTOption("Shrinkage="+str(Shrin_min))
	#self.setBDTOption("nCuts="+str(nCuts_min))
	test_max=0
	best_NT=0
	best_Sh=0
	best_nC=0
	ntrees=range(0,Schritte)
	shrin=range(0,Schritte)
	ncuts=range(0,Schritte)
	ijk=[0,0,0]
	
	mystyle=ROOT.gStyle.SetOptStat(0)

	timelist=[]
	ROClist=[]
	KSSlist=[]
	KSBlist=[]
	OPTlist=[]

	for i in range(0,Schritte):
	  ntrees[i]=NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte)
	  self.setBDTOption("NTrees="+str(NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte)))
	  for k in range(0,Schritte):
	    shrin[k]=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
	    self.setBDTOption("Shrinkage="+str(Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)))
	    k1=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
            time = self.trainBDT([],"")
	      #self.testBDT([],"")
	    ROC, ksS, ksB, ROCT = self.evaluateLastTraining()
            #roc_hist.SetBinContent(k+1,i+1,ROC)
	    #roct_hist.SetBinContent(k+1,i+1,ROCT)
	    #ratio_hist.SetBinContent(k+1,i+1,(ROC/ROCT))
	    timelist.append(time)
	    ROClist.append(ROC)
	    KSSlist.append(ksS)
	    KSBlist.append(ksB)
	    OPTlist.append(self.bdtoptions)

        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        resultfile='TMVA_resultfile_'+dt+'.csv'
        results = open(resultfile, "a+")
        for i in range(len(timelist)):
            results.write(str(timelist[i])+',best AMS = ,'+str(self.bookbetterReader())+','+str(ROClist[i])+','+str(KSSlist[i])+','+str(KSBlist[i])+','+str(OPTlist[i])+'\n')
        results.close()    

       
    def ams(x, y, w, cut):
    # Calculate Average Mean Significane as defined in ATLAS paper
    #    -  approximative formula for large statistics with regularisation
    # x: array of truth values (1 if signal)
    # y: array of classifier result
    # w: array of event weights
    # cut
        #t = y > cut 
        #s = np.sum((x[t] == 1)*w[t])
        #b = np.sum((x[t] == 0)*w[t])
        #return s/np.sqrt(b+10.0)
        s=0
        b=0
        for t in range(len(y)):
            if y[t] > cut:
	       if (x[t] == 1):
                s += 1*w[t]
               if (x[t] == 0):
                b += 1*w[t]
        return s/np.sqrt(b+10.0)
    
    def find_best_ams(self, x, y, w):
    # find best value of AMS by scanning cut values; 
    # x: array of truth values (1 if signal)
    # y: array of classifier results
    # w: array of event weights
    #  returns 
    #   ntuple of best value of AMS and the corresponding cut value
    #   list with corresponding pairs (ams, cut) 
    # ----------------------------------------------------------
        ymin=np.min(y) # classifiers may not be in range [0.,1.]
        print 'ymin = ', ymin
        ymax=np.max(y)
        print 'ymax = ' ,ymax
        #print x, y, w
        print type(x), type(y), type(w)
        nprobe=200    # number of (equally spaced) scan points to probe classifier 
        cuts = np.linspace(ymin, ymax, nprobe)
        #print cuts
        y=y.tolist()
        amsvec=[]
        for cut in cuts:
	  amsvec.append((self.ams(x, y, w, cut), cut))
        maxams=sorted(amsvec, key=lambda lst: lst[0] )[-1]
        #return maxams, amsvec

	#maxams_BDT, amsvec_BDT=find_best_ams(x, y, w)

	print "Average Mean Sensitivity (AMS) and cut value:"
	print maxams
	return maxams
      
      
    def bookReader(self,variables_=[],bdtoptions_="",factoryoptions_=""):#, weightfile):
	ROOT.gStyle.SetOptStat(0)
	localvar=[]
	if not hasattr(self, 'signal_train') or not hasattr(self, 'signal_test') or not hasattr(self, 'background_train')  or not hasattr(self, 'background_test'):
            print 'set training and test samples first'
            return
	newbdtoptions=replaceOptions(bdtoptions_,self.bdtoptions)
        newoptions=replaceOptions(factoryoptions_,self.factoryoptions)
        reader = ROOT.TMVA.Reader(newoptions)
        variables=variables_
        if len(variables)==0:
            variables = self.best_variables
        for i in range(len(variables)):
	    localvar.append(array('f',[0]))
	    exec('var_'+str(i)+" = array('f',[0])") in globals(), locals()
	    #eval("var_"+str(i))=array('f',[0])
            #reader.AddVariable(variables[i],eval("var_"+str(i)))
            reader.AddVariable(variables[i],localvar[i])
            
        # book reader
        reader.BookMVA( "BDTG", self.trainedweight)#'weights/weights_2016_0613_142658.xml')#self.trainedweight)
        
        # add signal and background trees
        input_test_S = ROOT.TFile( self.signal_test.path )
        input_test_B = ROOT.TFile( self.background_test.path )          
        test_treeS = input_test_S.Get(self.streename)
        test_treeB = input_test_B.Get(self.btreename)
	ROOT.gROOT.SetBatch(True)
    
	
	c = ROOT.TCanvas('c','c',800,600)
	sighisto = ROOT.TH1F('sighisto','sighisto',50,-2,2)
	bkghisto = ROOT.TH1F('bkghisto','bkghisto',50,-2,2)
	
	#loop over events in trees and save BDToutput
	for evt in test_treeS:
            for i in range(len(localvar)):  
                exec("localvar[i]=test_treeS."+str(variables[i]))
            y = reader.EvaluateMVA( "BDTG" )
            self.signal_prediction.append(y)
            sighisto.Fill(y)
	  
	for evt in test_treeB:
            for i in range(len(localvar)):
                exec("localvar[i]=test_treeB."+str(variables[i]))
            y = reader.EvaluateMVA( "BDTG" )
            self.background_prediction.append(y)
            bkghisto.Fill(y)
	
	sighisto.Draw()
	bkghisto.Draw("SAME")
	c.SaveAs('../../outhisto.pdf')
	
	
	#mvaValue = reader.EvaluateMVA( "BDTG" )
	##print mvaValue
	#varx[0] = -1
	#vary[0] = 1
	#we = reader.EvaluateMVA("BDTG")
	##print we
	## create a new 2D histogram with fine binning
	#histo2 = ROOT.TH2F("histo2","",200,-5,5,200,-5,5)
	#maxout=0
	#minout=0
	
	
    def bookbetterReader(self):
      feature_file = ROOT.TFile(self.treepath, 'read')
      feature_tree = feature_file.Get("validation")
      tmva_reader = ROOT.TMVA.Reader("!Color:!Silent")
      
      feature_values = []
      for variable in self.best_variables:
	feature_values.append(array( 'f',  [0.]))
	feature_tree.SetBranchAddress(variable,  feature_values[-1])
	tmva_reader.AddVariable(variable, feature_values[-1])
      
      label_value = array( 'c',  ['x'])
      weight_value = array( 'f',  [0.0])
      
      feature_tree.SetBranchAddress('Label',  label_value)
      feature_tree.SetBranchAddress('Weight',  weight_value)
      
      tmva_reader.BookMVA( "BDTG", self.trainedweight)
      
      result = np.zeros((feature_tree.GetEntries(), 3))
      
      for ievt in range(feature_tree.GetEntries()):
	feature_tree.GetEntry(ievt)
	result[ievt, 0] = int(label_value[0] == 's')
	result[ievt, 1] = weight_value[0]
	result[ievt, 2]    = tmva_reader.EvaluateMVA('BDTG')
	
      best_ams, vec = self.find_best_ams(result[:,0],result[:,2],result[:,1])
      print best_ams
      return best_ams
      
	
    def ams(self, x, y, w, cut):

    # Calculate Average Mean Significane as defined in ATLAS paper

    #    -  approximative formula for large statistics with regularisation

    # x: array of truth values (1 if signal)

    # y: array of classifier result

    # w: array of event weights

    # cut

        t = y > cut 

        s = np.sum((x[t] == 1)*w[t])

        b = np.sum((x[t] == 0)*w[t])

        return s/np.sqrt(b+10.0)
	
    def find_best_ams(self, x, y, w):

    # find best value of AMS by scanning cut values; 

    # x: array of truth values (1 if signal)

    # y: array of classifier results

    # w: array of event weights

    #  returns 

    #   ntuple of best value of AMS and the corresponding cut value

    #   list with corresponding pairs (ams, cut) 

    # ----------------------------------------------------------

        ymin=min(y) # classifiers may not be in range [0.,1.]

        ymax=max(y)

        nprobe=200    # number of (equally spaced) scan points to probe classifier 

        amsvec= [(self.ams(x, y, w, cut), cut) for cut in np.linspace(ymin, ymax, nprobe)] 

        maxams=sorted(amsvec, key=lambda lst: lst[0] )[-1]

        return maxams, amsvec      