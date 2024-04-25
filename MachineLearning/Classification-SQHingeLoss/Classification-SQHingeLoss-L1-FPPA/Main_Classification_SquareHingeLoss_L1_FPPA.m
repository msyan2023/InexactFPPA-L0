

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L1 norm regularization)                   
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model   
%
%    min_u  lambda*||u||_1 + psi(Bu)  
%    where  psi(x) = 1/2*max(1-x,0)^2    square hinge loss
%                B = diag(TrainLabel)*KernelMat


clc, clear, close all
restoredefaultpath 
addpath('Data')
disp('Classification-Square Hinge Loss-FPPA')
disp('----------------------------------------------')
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

DataTrain= ImgsTrain(:,1:1:5000);
LabelTrain = LabelsTrain(1:5000);
DataTest = ImgsTest(:,1:2000);
LabelTest = LabelsTest(1:2000);

NumTrain = length(LabelTrain);
NumTest = length(LabelsTest);

global MatB
%% Kernel Parameter
paraModel.sigma = 4;  %% Gaussian kernel parameter

%% Generate Kernel Matrix 
KernelMat = GaussKernel(DataTrain, DataTrain, paraModel.sigma);
MatB = diag(LabelTrain)*KernelMat;
%% Generate Prediction Matrix
PredMat = GaussKernel(DataTrain,DataTest, paraModel.sigma);

%% choose parameter in proximity algorithm
paraModel.lambda = 1;   %% regularization parameter

paraFPPA.MaxIter = 5e4;  %% maximal iteration in FPPA   
paraFPPA.rho = 500; 
paraFPPA.beta = (1+1e-6)*norm(MatB,2)^2/paraFPPA.rho;    %% convergence condition 
paraFPPA.RelStop = 1e-4;
paraFPPA.NumShow = 100;

paraModel
paraFPPA
%%  Training
[u, v, History] = Classification_SquareHingeLoss_L1_FPPA(KernelMat, PredMat, LabelTrain, LabelTest, paraFPPA, paraModel); 

fname = sprintf('Classification_L1_Test%0.2f_NNZ%d_Train%0.2f_lambda%0.2e_rho%0.2e.mat',...
    History.TestAccuracy(end), History.NNZ(end), History.TrainAccuracy(end), paraModel.lambda, paraFPPA.rho)
save(fname, 'u', 'paraFPPA', 'paraModel')

%% show result
fprintf('Number of Nonzeros of u: %d ------(length of u=%d)\n', History.NNZ(end), NumTrain)
fprintf('Training accuracy: %0.2f%% ------(%d training samples totally)\n', History.TrainAccuracy(end), NumTrain)
fprintf('Testing accuracy: %0.2f%% ------(%d testing samples totally)\n', History.TestAccuracy(end), NumTest)


 
