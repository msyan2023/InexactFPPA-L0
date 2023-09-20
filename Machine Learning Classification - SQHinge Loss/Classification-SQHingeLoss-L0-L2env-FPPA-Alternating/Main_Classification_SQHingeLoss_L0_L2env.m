

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L0 norm regularization)                   L2 envelop   22222222222222222222222
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model
%
%    min_{u,v} lambda*||u||_0 + lambda/gamma*1/2*||u-v||_2^2 + psi(Bv)
%    where  psi(x) = 1/2*max(1-x,0)^2    square hinge loss
%                B = diag(TrainLabel)*KernelMat


clear, clc, close all
restoredefaultpath
addpath('Data')

format short
dbstop if error
%% Goal: binary classification of handwritten digit numbers: $num1$ and $num2$
num1 = 7;
num2 = 9;
fprintf('Handwritten Digit Recognization: %d and %d\n\n', num1, num2)

load ImgsTrain.mat
load LabelsTrain.mat
load ImgsTest.mat
load LabelsTest.mat



global MatB


DataTrain= ImgsTrain(:,1:1:5000);
LabelTrain = LabelsTrain(1:5000);
DataTest = ImgsTest(:,1:2000);
LabelTest = LabelsTest(1:2000);

NumTrain = length(LabelTrain);
NumTest = length(LabelsTest);

disp('Classification-Square Hinge Loss-L0 Regularization-L2 envelop-FPPA')
disp('Stopping Criteria of Inner Iteration: Sufficient Decreasing Condition (Alternating Minimization Problem)')
disp('---------------------------------------------------------------------------------')



%% Kernel Parameter
paraModel.sigma = 4;  %% Gaussian kernel parameter

%% Generate Kernel Matrix
KernelMat = GaussKernel(DataTrain, DataTrain, paraModel.sigma);
MatB = diag(LabelTrain)*KernelMat;
%% Generate Prediction Matrix
PredMat = GaussKernel(DataTrain,DataTest, paraModel.sigma);

%% choose parameter in proximity algorithm

% bigger lambda, smaller gamma --->  less NNZ

% model parameter
paraModel.lambda = 9e-3;   %% regularization parameter
paraModel.gamma = 1e-3;  %%  envelop parameter
 
%%%%%%%%%%%%%%%%%   Use initial value = 0 %%%%%%%%%%%%%%%%%%

%----- Outer iteration parameter
paraAlgo.Out.MaxIter = 5e4;  %% maximal iteration for Outer iteration
paraAlgo.Out.alpha = 0.99;  %% average parameter    (u2 = prox{ (1-alpha)*u1 + alpha*v1 }    alpha \in (0,1)
paraAlgo.Out.theta = 1;   %% average parameter    (v2 = prox{ theta*u2 + (1-theta)*v1 }   theta \in (0,1]
paraAlgo.Out.RelStop = 1e-4;
paraAlgo.Out.NumShow = 10;

%----- Inner iteration parameter
paraAlgo.In.rho = 1;
paraAlgo.In.beta = (1+1e-6)*norm(MatB,2)^2/paraAlgo.In.rho;   %% convergence condition in inner loop
paraAlgo.In.FacInnerSTOP = 0.99;  %%   factor involved in stopping criteria of inner iteration
paraAlgo.In.NormInnerSTOP = 1e16; %% summable condition involved in stopping criteria of inner iteration


paraModel
paraAlgo
paraAlgo.Out
paraAlgo.In
fprintf('Threshold of ell0 truncation = %f\n\n',sqrt(2*paraModel.gamma*paraAlgo.Out.alpha))
disp('-----------------------------------------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Training

[u, v, w, History] = Alternating_SQHingeLoss_L0_L2env(KernelMat, PredMat, LabelTrain, LabelTest, paraAlgo, paraModel);

fname = sprintf('Classification_L0_Test%0.2f_NNZ%d_Train%0.2f_lambda%.2e_gamma%.2e_rho%.0e.mat',...
    History.AccuracyTest(end), History.NNZ(end), History.AccuracyTrain(end), paraModel.lambda, paraModel.gamma, paraAlgo.In.rho)
save(fname, 'u', 'paraModel', 'paraAlgo')

%% show result
fprintf('Number of Nonzeros of u: %d ------(length of u=%d)\n', History.NNZ(end), NumTrain)
fprintf('Training accuracy: %0.2f%% ------(%d training samples totally)\n', History.AccuracyTrain(end), NumTrain)
fprintf('Testing accuracy: %0.2f%% ------(%d testing samples totally)\n', History.AccuracyTest(end), NumTest)



