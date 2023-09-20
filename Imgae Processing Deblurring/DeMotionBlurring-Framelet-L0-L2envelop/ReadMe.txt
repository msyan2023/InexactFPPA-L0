
09-19-2023

Implement Algorithms in the paper to solve the image deblurring L0-Framelet model. 

---- Regular stopping criteria for inner iteration (Algorithm 1 in the paper) ----
	
	In the file "DeMotionBlurring_Framelet_L0_L2env_FPPA.m", make sure using 

	    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(theta*gamma/lambda, temp_v, u2, v1, w1, paraAlgo, paraModel, Img_obs, InnerTOL, k);
	
	Run "Main_DeMotionBlurring_Framelet_L0_L2env_SquareLoss.m"

---- 1-step inner iteration (First variant of Algorithm 1) ----

	In the file "DeMotionBlurring_Framelet_L0_L2env_FPPA.m", make sure using 

	    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B_1stepInnerIteration(theta*gamma/lambda, temp_v, u2, v1, w1, paraAlgo, paraModel, Img_obs, InnerTOL, k);
	
	Run "Main_DeMotionBlurring_Framelet_L0_L2env_SquareLoss.m"

 
