

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L1 norm regularization)          Machine Learning Regression          
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model   
%
%    min_u  lambda*||u||_1 + psi(Bu)   
%    where  psi(x) = 1/2*||x-y||^2    square loss where y is given training label
%                     B = KernelMat

restoredefaultpath 
addpath('Data')
format short
dbstop if error

disp('Regression-Square Loss-L1 Regularization-FPPA')
disp('---------------------------------------------------------------------------------')


load MgDataLabel.mat


global MatB

NumTrain = 1000;
NumTest = 385;

DataTrain= MgData(:,1:1:NumTrain);
LabelTrain = MgLabel(1:NumTrain);
DataTest = MgData(:,NumTrain+1:1:end);
LabelTest = MgLabel(NumTrain+1:end);

 
%% Kernel Parameter
paraModel.sigma = sqrt(10);  %% Gaussian kernel parameter

%% Generate Kernel Matrix 
KernelMat = GaussKernel(DataTrain, DataTrain, paraModel.sigma);
MatB = KernelMat;
%% Generate Prediction Matrix
PredMat = GaussKernel(DataTrain,DataTest, paraModel.sigma);

%% choose parameter in proximity algorithm
paraModel.lambda = 0.1;   %% regularization parameter

paraFPPA.MaxIter = 10e4;  %% maximal iteration in FPPA   
paraFPPA.rho = 100; 
paraFPPA.beta = norm(MatB,2)^2/paraFPPA.rho*(1+1e-6);    %% convergence condition 
paraFPPA.RelStop = 1e-6;
paraFPPA.NumShow = 1000;

paraModel
paraFPPA
%%  Training
[u, ~, History, StopIter] = Regression_SQLoss_L1_FPPA(KernelMat, PredMat, LabelTrain, LabelTest, paraFPPA, paraModel); 

fname = sprintf('Regression_L1_Test%0.5f_NNZ%d_Train%0.5f_lambda%0.2e_rho%0.2e.mat',...
    History.TestMSE(end), History.NNZ(end), History.TrainMSE(end), paraModel.lambda, paraFPPA.rho)
save(fname, 'u', 'paraFPPA', 'paraModel','StopIter')

%% show result
fprintf('Number of Nonzeros of u: %d ------(length of u=%d)\n', History.NNZ(end), NumTrain)
fprintf('Training MSE: %0.5f ------(%d training samples totally)\n', History.TrainMSE(end), NumTrain)
fprintf('Testing MSE: %0.5f ------(%d testing samples totally)\n', History.TestMSE(end), NumTest)

 
 
