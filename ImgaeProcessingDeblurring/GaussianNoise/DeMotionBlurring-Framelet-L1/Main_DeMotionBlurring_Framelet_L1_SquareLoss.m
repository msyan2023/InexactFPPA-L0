

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L1 norm regularization)  
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model   (x is given blurred and noisy image)
%
%    min_u 1/2*|| Ku - x ||_2^2 + lambda || Du ||_1
%
%               u is an image (matrix) and x is observed motion blurred noisy image
%
%               K is blurring kernel matrix (motion blurring with symmetric(reflexive/mirror) padding condition)
%                     the computation of prox_{psi\circ B} needs an inverse of a matrix and for this case of blurring,
%                     there is no fast algorthm(not like padding with periodic condition) provided, thus inner iteration
%                     for estimating prox_{1/2*|| Ku - x ||_2^2} is involved
%
%               D is DCT-induced tight framelet decomposition matrix
%

clear, clc, close all
format short
dbstop if error

restoredefaultpath
addpath('data')
addpath('GenerateTightFrameletFilter')
addpath('TightFramelet')
addpath('Motion-blurring')

disp('Deblurring-Square Loss-L1 Regularization-FPPA')
disp('Stopping Criteria of Inner Iteration:  Sumable condition')
disp('---------------------------------------------------------------------------------')

%% set up image
% Img_ref = imread('barbara.png'); Img_ref = Img_ref(1:256,257:end);
Img_ref = imread('airplane.gif');
[rows, cols] = size(Img_ref); Img_ref = double(Img_ref);
%% Blurring
paraModel.Blur.blur_length = 21;
paraModel.Blur.blur_angle = 45;
paraModel.Blur.blur_filter = fspecial('motion',paraModel.Blur.blur_length,paraModel.Blur.blur_angle);   %% motion blurring filter

Img_blur = imfilter(Img_ref, paraModel.Blur.blur_filter, 'symmetric');  %% blur with symmetric(mirror) padding
%% Adding Noise
paraModel.NoiseLevel = 3; % the standard deviation of the Gaussian noise.
rng('default')
Img_obs = Img_blur + paraModel.NoiseLevel*randn(rows,cols);
Img_obs = double(Img_obs);
%% choose parameter of tight framelet induced by DCT
paraModel.Framelet.J = 3;
[paraModel.Framelet.DecFil, paraModel.Framelet.RecFil] = Generate_OneD_Filter(paraModel.Framelet.J);

%% choose parameter in proximity algorithm

% model parameter
paraModel.lambda = 0.18;   %% regularization parameter

%%%%%%%%%%%%%%%%%   Use initial value = 0 %%%%%%%%%%%%%%%%%%

%----- Outer iteration parameter
% paraAlgo.Out.rho --> parameter p1 in the paper 
% paraAlgo.Out.beta --> parameter q1 in the paper
paraAlgo.Out.MaxIter = 5000;  %% maximal iteration for Outer iteration
paraAlgo.Out.RelStop = 1e-5;
paraAlgo.Out.rho = 0.01;   %% this is the parameter p1 in the paper 
paraAlgo.Out.beta = (1+1e-6) * 1/paraAlgo.Out.rho;   %% convergence condition: beta*rho>||framelet matrix||_2^2=1
 
%----- Inner iteration parameter
% paraAlgo.In.rho --> parameter p2 in the paper
% paraAlgo.In.beta --> parameter q2 in the paper
paraAlgo.In.rho = 0.2;   
paraAlgo.In.beta = (1+1e-6) * 4/paraAlgo.In.rho;   %% convergence condition: beta*rho>||blurring kernel matrix||_2^2
paraAlgo.In.NormInnerSTOP = 1e8; %% summable condition involved in stopping criteria of inner iteration

paraModel
paraAlgo
paraAlgo.Out
paraAlgo.In
disp('-----------------------------------------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Training
tic
[u, v, History] = DeMotionBlurring_Framelet_L1_FPPA(paraAlgo, paraModel, Img_obs, Img_ref);
time = toc;
fname = sprintf('L1-DeMotionBlurring_PSNR%2.5f_lambda%2.2f_OUTrho%2.2f_INrho%2.2f.mat',...
    History.PSNR(end), paraModel.lambda, paraAlgo.Out.rho, paraAlgo.In.rho)
save(fname, 'u', 'v', 'History', 'paraModel', 'paraAlgo')


%% show result
figure, imshow(uint8(Img_ref)), title('Original Image')
figure, imshow(uint8(Img_blur)), title('Blurred Image')
figure, imshow(uint8(Img_obs)), title('Observed Image')
figure, imshow(uint8(u)), title('Restored Image')

figure, plot(History.PSNR), title('PSNR'), xlabel('Iteration')
figure, plot(History.TargetValue), title('TargetValue'), xlabel('Iteration')
figure, plot(History.InnerNum), title('InnerNum'), xlabel('Iteration')


fprintf('Time: %f s\n', time)
fprintf('PSNR: %f (before restoration)\n',mypsnr(Img_ref,Img_obs))
fprintf('PSNR: %f (after restoration)\n',History.PSNR(end))
