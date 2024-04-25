
09-19-2023

Implement Algorithms in the paper to solve the L0-SVM classification model. 

---- Regular stopping criteria for inner iteration (Algorithm 1 in the paper) ----
	
	In the file "Alternating_SQHingeLoss_L0_L2env.m", make sure using 

	    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(theta*gamma/lambda, theta*u2+(1-theta)*v1, LabelTrain, u2, v1, w1, paraAlgo, paraModel, InnerTOL, k);
	
	Run "Main_Classification_SQHingeLoss_L0_L2env.m"

---- 1-step inner iteration (First variant of Algorithm 1) ----

	In the file "Alternating_SQHingeLoss_L0_L2env.m", make sure using 

	    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B_1stepInnerIteration(theta*gamma/lambda, theta*u2+(1-theta)*v1, LabelTrain, u2, v1, w1, paraAlgo, paraModel, InnerTOL, k);	
	
	Run "Main_Classification_SQHingeLoss_L0_L2env.m"

---- Various rho prime (Second variant of Algorithm 1) ----

	In the file "Alternating_SQHingeLoss_L0_L2env.m", make sure using 

	    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(theta*gamma/lambda, theta*u2+(1-theta)*v1, LabelTrain, u2, v1, w1, paraAlgo, paraModel, InnerTOL, k);
	
	Modify the parameter "paraAlgo.In.FacInnerSTOP" in the main program with 10, 100, 1000 times 

	Run "Main_Classification_SQHingeLoss_L0_L2env.m"
