#code mostly taken from  6.2 of the Computerpraktikum Datenanlyse 2014
######################################################################


import numpy as np
from ROOT import *

def modified_rosenbrock_function(par,arr):
  a,b,c=arr[0],arr[1],arr[2]
  f = (par[0]**2+par[1]-a)**2+(par[0]+par[1]**2-b)**2+c*(par[0]+par[1])**2
  return f
  
  
  # Code fragment for exercise 6.2 of the Computerpraktikum Datenanlyse 2014
  # Authors: Ralf Ulrich, Frank Schroeder (Karlsruhe Institute of Technology)
  # This code fragment probably is not the best and fastest implementation
  # for "simulated annealing", but it is a simple implementation which does its job.  
def simAnnealing(a,b,c):
  nParameter = 2 # 2 parameters: x and y
  listOfPoints=[]


#just for testing on random seeds...  
  #tr = TRandom()
  #print tr.GetSeed()
  #tr1 = TRandom1()
  #print tr1.GetSeed()
  #tr2 = TRandom2()
  #print tr2.GetSeed()
  #tr3 = TRandom3()
  #print tr3.GetSeed()
  
  ##gRandom.SetSeed(0)
  #print gRandom.GetSeed()
  
  # Starting point: test the dependence of the algorithm on the initial values
  initialXvalue = 3
  initialYvalue = -2

  # Parameters of the  algorithm:
  # Find a useful set of parameters which allows to determine the global
  # minimum of the given function:
  # The temperature scale must be in adequate relation to the scale of the function values,
  # the step size must be in adequate relation to the scale of the distance between the 
  # different local minima
  initialTemperature = 10000
  finalTemperature = 1
  coolingSpeed = 1 # in percent of current temerature --> defines number of iterations
  stepSize = 1

  # Current parameters and cost
  currentParameters  = [initialXvalue, initialYvalue] # x and y in our case
  currentFunctionValue =  modified_rosenbrock_function(currentParameters, [a,b,c]) # you have to implement the function first!
  
  # keep reference of best parameters
  bestParameters = currentParameters
  bestFunctionValue = currentFunctionValue
  
  # Heat the system
  temperature = initialTemperature
  
  iteration = 0
    
  # Start to slowly cool the system
  while (temperature > finalTemperature): 
    
    # Change parameters
    newParameters = [0]*nParameter
    
    for ipar in range(nParameter):
      newParameters[ipar] = gRandom.Gaus(currentParameters[ipar], stepSize)
    
    # Get the new value of the function
    newFunctionValue = modified_rosenbrock_function(newParameters, [a,b,c])
    
    # Compute Boltzman probability
    deltaFunctionValue = newFunctionValue - currentFunctionValue
    saProbability = np.exp(-deltaFunctionValue / temperature)

    # Acceptance rules :
    # if newFunctionValue < currentFunctionValue then saProbability > 1
    # else accept the new state with a probability = saProbability
    if ( saProbability > gRandom.Uniform() ):
      currentParameters = newParameters
      currentFunctionValue = newFunctionValue
      listOfPoints.append(currentParameters)  # log keeping: keep track of path
      
    if (currentFunctionValue < bestFunctionValue):
      bestFunctionValue = currentFunctionValue
      bestParameters = currentParameters

    #print "T = ", temperature, "(x,y) = ",currentParameters, " Current value: ", currentFunctionValue, " delta = ", deltaFunctionValue # debug output
    
    # Cool the system
    temperature *= 1 - coolingSpeed/100.
    iteration+=1  # count iterations
    
  # end of cooling loop
 
 
  xResult = bestParameters[0]
  yResult = bestParameters[1]
  print iteration
  return xResult, yResult

x,y = simAnnealing(11,7,0.1)
print 'x_min = ',x
print 'y_min = ',y
val=[x,y]
print 'function_value = ', modified_rosenbrock_function(val,[11,7,0.1])
print 'val of mathematica = ', modified_rosenbrock_function([3,2],[11,7,0.1])
print 'val of plot_min = ', modified_rosenbrock_function([-4,-3],[11,7,0.1])

