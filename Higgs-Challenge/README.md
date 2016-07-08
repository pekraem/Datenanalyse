-----------------------------------------------------------
-----     short instruction into training skripts     -----
-----------------------------------------------------------

run training:
python test.py


change BDT-options:
in test.py:
	--> trainer.setBDTOption("option=value")	#setBDToption(string)


select variables:
in test.py:
	--> variables=[]	#contains all variables, remove unwanted ones
