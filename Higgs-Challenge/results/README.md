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
