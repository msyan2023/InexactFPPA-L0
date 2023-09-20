

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L0 norm regularization)                   L2 envelop   22222222222222222222222
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model
%
%    min_{u,v} lambda*||u||_0 + lambda/gamma*1/2*||u-v||_2^2 + psi(Bv)
%    where  psi(x) = 1/2*||x-y||^2    square loss where y is given training label
%                     B = KernelMat

restoredefaultpath
addpath('Data')
format short
dbstop if error

disp('Regression-Square Loss-L0 Regularization-L2 envelop-FPPA')
disp('Stopping Criteria of Inner Iteration: Sufficient Decreasing Condition + Summable Condition (Alternating Minimization Problem)')
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

% bigger lambda, smaller gamma --->  less NNZ

% model parameter
paraModel.lambda = 3.5e-6;   %% regularization parameter
paraModel.gamma = 1e-5;  %%  envelop parameter
 
%%%%%%%%%%%%%%%%%   Use initial value = 0 %%%%%%%%%%%%%%%%%%

%----- Outer iteration parameter
paraAlgo.Out.MaxIter = 10e4;  %% maximal iteration for Outer iteration
paraAlgo.Out.alpha = 0.99;  %% average parameter    (u2 = prox{ (1-alpha)*u1 + alpha*v1 }    alpha \in (0,1)
paraAlgo.Out.theta = 1;   %% average parameter    (v2 = prox{ theta*u2 + (1-theta)*v1 }   theta \in (0,1]
paraAlgo.Out.RelStop = 1e-6;
paraAlgo.Out.NumShow = 1000;

%----- Inner iteration parameter
paraAlgo.In.rho = 10;
paraAlgo.In.beta = norm(MatB,2)^2/paraAlgo.In.rho*(1+1e-6);   %% convergence condition in inner loop

paraAlgo.In.FacInnerSTOP = 0.99;  %% factor involved in stopping criteria of inner iteration
% 'rho prime' in the paper is linear about paraAlgo.In.FacInnerSTOP

paraAlgo.In.NormInnerSTOP = 1e16; %% summable condition involved in stopping criteria of inner iteration


paraModel
paraAlgo
paraAlgo.Out
paraAlgo.In
fprintf('Threshold of ell0 truncation = %f\n\n',sqrt(2*paraModel.gamma*paraAlgo.Out.alpha))
disp('-----------------------------------------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Training

[u, ~, ~, History,StopIter] = Alternating_SQLoss_L0_L2env(KernelMat, PredMat, LabelTrain, LabelTest, paraAlgo, paraModel);
 
fname = sprintf('Regression_L0_Test%0.5f_NNZ%d_Train%0.5f_lambda%.2e_gamma%.2e_rho%.0e.mat',...
    History.TestMSE(end), History.NNZ(end), History.TrainMSE(end), paraModel.lambda, paraModel.gamma, paraAlgo.In.rho)
save(fname, 'u', 'paraModel', 'paraAlgo','StopIter')

%% show result
fprintf('Number of Nonzeros of u: %d ------(length of u=%d)\n', History.NNZ(end), NumTrain)
fprintf('Training MSE: %0.5f ------(%d training samples totally)\n', History.TrainMSE(end), NumTrain)
fprintf('Testing MSE: %0.5f ------(%d testing samples totally)\n', History.TestMSE(end), NumTest)

 
