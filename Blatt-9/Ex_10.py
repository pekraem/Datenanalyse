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
  

#-----------------------------------------------------------------
class ROOT_Likelihood:
  #def __init__(self, signal_data, bckgrd_data):
        #self.signal_data = signal_data
        #self.bckgrd_data = bckgrd_data
        #self.data = np.vstack([signal_data, bckgrd_data])
        #self.clf = 0

  def fit(self):
    signal2d = TH2F("signal2d","",20,0,10,20,0,10)
    c = TCanvas("c","c",800,600)
    #c.Print("../../Ex_10-2_hist2d_setosa_prob.pdf[")
    gStyle.SetOptStat(0)
    c.Close()
    nvar=4
    nbins=5

    for i in range(nvar):
      for j in range(nvar):
	c = TCanvas("c","c",800,600)
	minx=min(data[:,i])
	maxx=max(data[:,i])
	miny=min(data[:,j])
	maxy=max(data[:,j])
	if j!=i:
	  signal2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	  background2d = TH2F("signal2d","",nbins,minx,maxx,nbins,miny,maxy)
	  for event in data:
	    if event[4]==2:
	      signal2d.Fill(event[i],event[j])
	    else:
	      background2d.Fill(event[i],event[j])  
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
    print x[0],x[1]
    print 'Bin = ',self.clf.FindBin(x[0],x[1])
    if self.clf == 0:
      print 'Fit Classifier first!'
      return 0
    else:
      if self.clf.GetBinContent(self.clf.FindBin(x[0],x[1]))!=0.0:
	print self.clf.GetBinContent(self.clf.FindBin(x[0],x[1]))
      print self.clf.GetEntries()
      return self.clf.GetBinContent(self.clf.FindBin(x[0],x[1]))

    
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
cut = ROOT_Likelihood()
cut.fit()

# initialize Plotter Class
plotter = Plotter(data[signal, :ndim], data[bckgrd, :ndim])
# and amke plots 
plotter.plot_contour(cut)
plotter.plot_test_statistic(cut)
plotter.plot_roc(cut)
nvar=4
nbins=20
