#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------ 
"""
.. module:: ex10_template
    :synopsis example for discriminant analysis 

.. moduleauthor Thomas Keck
"""
# ------------------------------------------------------------------------ 
# useful imports 

import bisect
import numpy as np
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams['backend']='TkAgg'
from ROOT import *
import math

from matplotlib import pyplot as plt

### ------- load the Data set --------------------------------------------
# Load iris data
data = np.loadtxt('../../iris.data')

# Define dictionary with columns names: s=sepal, p=petal, l=lenght, w=width,
columns = {0: 'L(Kelch)', 1: 'W(Kelch)', 
           2: 'L(Blatt)', 3: 'W(Blatt)', 4: 'class'}

# Define boolean arrays corresponding to the three 
#      classes setosa, versicolor and virginica
setosa = data[:, 4] == 0
versicolor = data[:, 4] == 1
virginica = data[:, 4] == 2

# Signal is versicolor (can be changed to setosa or virginica)
signal = versicolor 
bckgrd = ~signal
# !! note: see indexing of arrays with boolenan array in pyhthon documentation 
#
# exmaples how to access the data:
#    data[signal]     # All events classified as signal
#    data[background] # All events classified as background
#    data[setosa]     # All events classified as setosa
#    data[versicolor | virginica] # All events classified as versicolor or virginica
#    data[:, :2]      # The first two columns (sepal length and sepal width) of all events
#    data[signal, :2] # The first two columns of the signal events
#    data[background, 2:4] # The 3. and 4. column (petal length and petal width) of the background events
#    data[:, 4]       # The label column of all events
#             (see the numpy documentation for further examples)

### ------- helper functions ----------------------------------
# creates List with varpairs for 2d-plotting
def permuteVars(variables):
	combs = itertools.combinations(variables, 2)
	varcombs = list(combs)
	indexcombs = []
	for pair in varcombs:
		indexpair = []
		for var in pair:
			indexpair.append(variables.index(var))
		indexcombs.append(indexpair)
	return varcombs, indexcombs



class Plotter(object):
    """
        class to display and evaluate the performance of a test-statistic 
    """
    def __init__(self, signal_data, bckgrd_data):
        self.signal_data = signal_data
        self.bckgrd_data = bckgrd_data
        self.data = np.vstack([signal_data, bckgrd_data])

    def plot_contour(self, classifier):
        # 1st variable as x-dimension in the plots
        xdim = 0
        # and 2nd variable as y-dimension
        ydim = 1
        # Draw the scatter-plots of signal and background 
        plt.scatter(self.signal_data[:, xdim], self.signal_data[:, ydim], 
          c='r', label='Signal')
        plt.scatter(self.bckgrd_data[:, xdim], self.bckgrd_data[:, ydim], 
          c='b', label='Background')

        # Evaluate the response function on a two-dimensional grid ...
        #   ... using the mean-values of the data for the remaining dimensions.
        xs = np.arange(min(self.data[:, xdim])-1, max(self.data[:, xdim])+1, 0.1)
        ys = np.arange(min(self.data[:, ydim])-1, max(self.data[:, ydim])+1, 0.1)

        means = np.mean(self.data, axis = 0) # calculate mean of each column
        responses = np.zeros((len(ys), len(xs)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                values = np.copy(means)
                values[xdim] = x
                values[ydim] = y
                responses[j, i] = float(classifier.evaluate(values))

        # Draw a contour plot
        X, Y = np.meshgrid(xs, ys)
        c=plt.contourf(X, Y, responses, alpha=0.5, cmap=plt.cm.coolwarm)
        cbar=plt.colorbar(c, orientation='vertical')
        # add the direction of the fisher vector (if specified)
        if hasattr(classifier, 'fisher_vector'):
            vector = classifier.fisher_vector / np.linalg.norm(classifier.fisher_vector)
            plt.axes().set_aspect('equal', 'datalim')
            plt.plot([-vector[xdim]+means[xdim], vector[xdim]+means[xdim]],
                          [-vector[ydim]+means[ydim], vector[ydim]+means[ydim]],
                          'k-', lw=4, label="Fisher Projection")
        plt.title(
          "scatter plot {} vs {} and classifier contour".format(
          columns[xdim], columns[ydim] ) )
        #cbar.draw_all()
        plt.show()

    def plot_test_statistic(self, classifier):
        # Draw Distribution of the test-statistic
        ns, binss, _ = plt.hist(map(classifier.evaluate, self.signal_data), 
          color='r', alpha=0.5, label='Signal' )
        nb, binsb, _ = plt.hist( map(classifier.evaluate, self.bckgrd_data), 
           color='b', alpha=0.5, label='Background' )
	plt.legend()
        plt.title("test statistic")
        plt.show()

    # calculate efficiencies and plot ROC-curves
    def plot_roc(self, classifier):
        ns, binss = np.histogram(map(classifier.evaluate, self.signal_data)) 
        nb, binsb = np.histogram(map(classifier.evaluate, self.bckgrd_data)) 
        # enforce common binning for response on bkg and sig
        minresp=min([ binss[0], binsb[0] ])
        maxresp=max([ binss[len(binss)-1], binsb[len(binsb)-1] ])
        nbins=100
        bins=np.linspace(minresp, maxresp, nbins) 
        bwid=(maxresp-minresp)/nbins
        # calculate cumulative distributions (i.e. bkg and sig efficiencies)
        h, b = np.histogram( map(classifier.evaluate, self.signal_data), bins, density=True)
        ns = np.cumsum(h)*bwid
        h, b = np.histogram( map(classifier.evaluate, self.bckgrd_data), bins, density=True) 
        nb = np.cumsum(h)*bwid
        # finally, draw bkg-eff vs. sig-eff
        f2, ax = plt.subplots(1, 1)
        ax.plot(1.-ns, nb, 'r-', 1.-ns, nb, 'bo', linewidth=2.0)
        ax.set_xlabel("signal efficiency")
        ax.set_ylabel("background rejection")
        ax.set_title("ROC curve")
        plt.show()

def find_bin(x, edges):
    """
        returns the bin number (in array of bin edges) corresponding to x 
        @param x: value for which to find correspoding bin number
        @param edges: array of bin edges
    """
    print edges
    print x
    print bisect.bisect(edges, x)
    return max(min(bisect.bisect(edges, x) - 1, len(edges) - 2), 0)


### ------- simple example: Classifier and usage with Plotter class ---------

# template of Classifier Class
class CutClassifier(object):
    """
        template implementation of a Classifier Class
    """
    def fit(self, signal_data, bckgrd_data):
        """ 
            set up classifier ("training")
        """
    # some examples of what might be useful:
      # 1. signal and background histograms with same binning
        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

      # 2. mean and covariance matrix 
        self.signal_mean = np.mean(signal_data, axis=0)
        self.signal_cov = np.cov(signal_data.T)
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)
        self.bckgrd_cov = np.cov(bckgrd_data.T)
        
      # 3. print stuff
        print 'signal_mean = ', self.signal_mean
        print 'background_mean = ', self.bckgrd_mean
        

    def evaluate(self, x):
    # simple example of a cut-base classifier
        c=0
        #print 'x = ',x
        for i in range(len(x)):
           c+=(x[i] < (self.signal_mean[i] + self.bckgrd_mean[i])/2.)     
        return c

class LogLikeliClassifier(object):
    """
        template implementation of a Classifier Class
    """
    def fit(self, signal_data, bckgrd_data):
        """ 
            set up classifier ("training")
        """
    # some examples of what might be useful:
      # 1. signal and background histograms with same binning
        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

      # 2. mean and covariance matrix 
        self.signal_mean = np.mean(signal_data, axis=0)
        self.signal_cov = np.cov(signal_data.T)
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)
        self.bckgrd_cov = np.cov(bckgrd_data.T)
        
      # 3. print stuff
        print 'signal_mean = ', self.signal_mean
        print 'background_mean = ', self.bckgrd_mean
        #self.signal_cov.print()
        #self.bckgrd_cov.print()
        
      # 4. create Histos with Gaussian distributions
	#for varpair in varpairs:
	  #c = TCanvas("c","c",800,600)
	nbins = 5  
	minx=min(data[:,0])
	maxx=max(data[:,0])
	miny=min(data[:,1])
	maxy=max(data[:,1])
	#if varpair[0]!=varpair[1]:
	signal2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	background2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	xs, ys = np.random.multivariate_normal(self.signal_mean, self.signal_cov, 5000).T
	xb, yb = np.random.multivariate_normal(self.bckgrd_mean, self.bckgrd_cov, 5000).T
	for x1,y1,x2,y2 in zip(xs,ys,xb,yb):
	  signal2d.Fill(x1,y1)
	  background2d.Fill(x2,y2)
	hist2dges = signal2d.Clone()
	hist2dges.Add(background2d)
	ratio = signal2d.Clone()
	ratio.Divide(background2d)
	ratio.Draw("colz")
	self.clf=ratio
	  #c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf")
	  #c.Close()
	#c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf]")
        self.signal_distribution = np.random.multivariate_normal(self.signal_mean, self.signal_cov, 5000).T
        self.bckgrd_distribution = np.random.multivariate_normal(self.bckgrd_mean, self.bckgrd_cov, 5000).T
        

    def evaluate(self,x):
      #print x[0],x[1]
      #print 'Bin = ',self.clf.FindBin(x[0],x[1])
      if self.clf == 0:
	print 'Fit Classifier first!'
	return 0
      else:
	binx = self.clf.GetXaxis().FindBin(x[0])
	biny = self.clf.GetYaxis().FindBin(x[1])
	bin = self.clf.GetBin(binx,biny,0)
	if bin not in bins:
	  bins.append(bin)
	val = self.clf.GetBinContent(bin)
	if val!=0:
	  print math.log(val)
	  return math.log(val)
	else:
	  print -9
	  return -9

class LikelihoodClassifier(object):
    """
        template implementation of a Classifier Class
    """
    def fit(self, signal_data, bckgrd_data):
        """ 
            set up classifier ("training")
        """
    # some examples of what might be useful:
      # 1. signal and background histograms with same binning
        _, self.edges = np.histogramdd(np.vstack([signal_data, bckgrd_data]), bins=10)
        self.signal_hist, _ = np.histogramdd(signal_data, bins=self.edges)
        self.bckgrd_hist, _ = np.histogramdd(bckgrd_data, bins=self.edges)

      # 2. mean and covariance matrix 
        self.signal_mean = np.mean(signal_data, axis=0)
        self.signal_cov = np.cov(signal_data.T)
        self.bckgrd_mean = np.mean(bckgrd_data, axis=0)
        self.bckgrd_cov = np.cov(bckgrd_data.T)
        
      # 3. print stuff
        print 'signal_mean = ', self.signal_mean
        print 'background_mean = ', self.bckgrd_mean

    def evaluate(self, x):
    # simple example of a cut-base classifier
        c=0
        print x
        #for i in range(len(x)):
        b = find_bin(x,self.edges)
        print 'Bin = ',b 
        c = 0
        return c


## example how to use a Classifier Class with the Plotter Class
#ndim = 2

## initialise Classifier with training data
#cut = CutClassifier()
#cut.fit(data[signal, :ndim], data[bckgrd, :ndim])

## initialize Plotter Class
#plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
## and amke plots 
#plotter.plot_contour(cut)
#plotter.plot_test_statistic(cut)
#plotter.plot_roc(cut)
#nvar=4
#nbins=20

# ------------------------------------------------------------------
def Exercise_1():
  plotlist=[]
  for i in range(nvar):
    for j in range(nvar):
      fig = plt.figure(figsize=(10,8))
      if i==j:
	plt.hist(data[signal,i], color='r', range=(min(data[:,i]),max(data[:,i])), bins=25, histtype='stepfilled', alpha=0.2, normed=False, label='signal')
	plt.hist(data[bckgrd,i], color='b', range=(min(data[:,i]),max(data[:,i])), bins=25, histtype='stepfilled', alpha=0.2, normed=False, label='background')
	plt.legend(loc='best')
      else:
	plt.scatter(data[:,i], data[:,j], c=data[:,4])
	plt.colorbar()
      plotlist.append(fig)
      plt.close()

  with PdfPages('../../Ex_10-1_scatterplots.pdf') as pdf:
    for plot in plotlist:
      pdf.savefig(plot)
  
bins=[]
#-----------------------------------------------------------------
class ROOT_Likelihood:
  #def __init__(self, signal_data, bckgrd_data):
        #self.signal_data = signal_data
        #self.bckgrd_data = bckgrd_data
        #self.data = np.vstack([signal_data, bckgrd_data])
        #self.clf = 0

  def fit(self,varpairs):
    signal2d = TH2F("signal2d","",20,0,10,20,0,10)
    c = TCanvas("c","c",800,600)
    #c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf[")
    gStyle.SetOptStat(0)
    c.Close()
    nvar=4
    nbins=4

    for varpair in varpairs:
	c = TCanvas("c","c",800,600)
	minx=min(data[:,varpair[0]])
	maxx=max(data[:,varpair[0]])
	miny=min(data[:,varpair[1]])
	maxy=max(data[:,varpair[1]])
	if varpair[0]!=varpair[1]:
	  signal2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	  background2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	  for event in data:
	    if event[4]==2:
	      signal2d.Fill(event[varpair[0]],event[varpair[1]])
	    else:
	      background2d.Fill(event[varpair[0]],event[varpair[1]])  
	  hist2dges = signal2d.Clone()
	  hist2dges.Add(background2d)
	  ratio = signal2d.Clone()
	  ratio.Divide(hist2dges)
	  ratio.Draw("colz")
	  self.clf=ratio
	  #c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf")
	  c.Close()
    #c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf]")
    
  def evaluate(self,x):
    #print x[0],x[1]
    #print 'Bin = ',self.clf.FindBin(x[0],x[1])
    if self.clf == 0:
      print 'Fit Classifier first!'
      return 0
    else:
      binx = self.clf.GetXaxis().FindBin(x[0])
      biny = self.clf.GetYaxis().FindBin(x[1])
      bin = self.clf.GetBin(binx,biny,0)
      if bin not in bins:
	bins.append(bin)
      print self.clf.GetBinContent(bin)
      return self.clf.GetBinContent(bin)

    
#Exercise_2_ROOT()    

#------------------------------------------------------------------
# Exercise 3


# ------------------------------------------------------------------
# Exercise 4


# ------------------------------------------------------------------
# Exercise 5
#   -> Set ndim to 4

# example how to use a Classifier Class with the Plotter Class
ndim = 2

# initialise Classifier with training data
cut = LogLikeliClassifier()
cut.fit(data[signal, :ndim], data[bckgrd, :ndim])

# initialize Plotter Class
plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
# and amke plots 
plotter.plot_contour(cut)
plotter.plot_test_statistic(cut)
plotter.plot_roc(cut)
nvar=4
nbins=20
