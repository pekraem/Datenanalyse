-----------------------------------------------------------------
-----     folder to save results of the Higgs-Challenge     -----
-----------------------------------------------------------------

all steps should be documented in this readme


-----------------------------------------------------------------
1. looking for reasonable variables:
	* correlationplots from TMVA outfile via plot_cor.py
	* plot variables with Plot_Exp.py
	* looking for good cuts in variables
	* looking for uncorrelated variables
	* TODO: removing bad variables
	* TODO: testing and comparing with new variable selection

2. implement training and testing:
	* use mvautils/trainer class by Hannes/Karim
	* changed skripts for Higgs sample
	* TODO: implement analysis part from template to compare ams score

3. implement some plotting functions

4. testing different MVAoptions, inputvariables for best score


-----------------------------------------------------------------
1. reasonable variables:
   * looked at scikit-learn feature_importances_
   * selected some variables

2. implementation:
   * scikit-learn scripts are (almost) running
   * testing some BDT options in scikit-learn --> overtraining
   * --> splitting sample in training and testing samples















------------------------------------------------------------------
scikit-learn training with big sample:

+++ Best AMS = 1.1326961838+++
with: learning_rate=0.014, n_estimators=1500, max_depth=5, random_state=0, loss=deviance,
subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
init=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=auto

