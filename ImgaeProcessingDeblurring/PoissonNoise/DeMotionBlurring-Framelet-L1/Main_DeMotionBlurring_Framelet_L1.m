

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (TightFramelet-L1 norm regularization)  
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model   (x is given (motion) blurred and (Poisson) noisy image)
%
%    min_u  varphi_x(Ku) + lambda || Du ||_1
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
%               For Gaussian noise, fidelity term is 
% 
%                                 varphi_x(z) = 1/2*||z-x||^2
% 
%               For Poisson noise, fidelity term is 
% 
%                                 varphi_x(z) = <z,1> + <ln(z),x> 
% 

clear, clc, close all
format short
dbstop if error

restoredefaultpath
addpath('data')
addpath('GenerateTightFrameletFilter')
addpath('TightFramelet')
addpath('Motion-blurring')

disp('Deblurring-L1 Regularization-FPPA')
disp('Stopping Criteria of Inner Iteration:  Sumable condition')
disp('---------------------------------------------------------------------------------')

%% set up image
% Img_ref = imread('barbara.png'); Img_ref = Img_ref(1:256,257:end);
Img_ref = imread('airplane.gif');
[rows, cols] = size(Img_ref); Img_ref = double(Img_ref);

%% special scale step for poisson noise 
paraModel.PoissonNoiseScale = 255;
Img_ref = Img_ref / max(Img_ref(:)) * paraModel.PoissonNoiseScale;

%% Blurring
paraModel.Blur.blur_length = 21;
paraModel.Blur.blur_angle = 45;
paraModel.Blur.blur_filter = fspecial('motion',paraModel.Blur.blur_length,paraModel.Blur.blur_angle);   %% motion blurring filter

Img_blur = imfilter(Img_ref, paraModel.Blur.blur_filter, 'symmetric');  %% blur with symmetric(mirror) padding

%% Adding Noise
rng('default')

% Gaussian Noise
% paraModel.NoiseLevel = 3; % the standard deviation of the Gaussian noise.
% Img_obs = Img_blur + paraModel.NoiseLevel*randn(rows,cols);
% Img_obs = double(Img_obs);

% Poisson Noise
Img_obs = poissrnd(Img_blur);
Img_obs = double(Img_obs);


imtool(Img_obs,[0 max(Img_obs(:))])

%% choose parameter of tight framelet induced by DCT
paraModel.Framelet.J = 3;
[paraModel.Framelet.DecFil, paraModel.Framelet.RecFil] = Generate_OneD_Filter(paraModel.Framelet.J);

%% choose parameter in proximity algorithm

% model parameter
paraModel.lambda = 0.03;   %% regularization parameter

%%%%%%%%%%%%%%%%%   Use initial value = 0 %%%%%%%%%%%%%%%%%%

%----- Outer iteration parameter
% paraAlgo.Out.rho --> parameter p1 in the paper 
% paraAlgo.Out.beta --> parameter q1 in the paper
paraAlgo.Out.MaxIter = 2000;  %% maximal iteration for Outer iteration
paraAlgo.Out.RelStop = 1e-5;
paraAlgo.Out.rho = 0.06;   %% this is the parameter p1 in the paper 
paraAlgo.Out.beta = (1+1e-6) * 1/paraAlgo.Out.rho;   %% convergence condition: beta*rho>||framelet matrix||_2^2=1
paraAlgo.Out.NumShow = 20;  %% number of iterations showing progress

%----- Inner iteration parameter
% paraAlgo.In.rho --> parameter p2 in the paper
% paraAlgo.In.beta --> parameter q2 in the paper
paraAlgo.In.rho = paraAlgo.Out.rho;   
paraAlgo.In.beta = (1+1e-6) * 4/paraAlgo.In.rho;   %% convergence condition: beta*rho>||blurring kernel matrix||_2^2
paraAlgo.In.NormInnerSTOP = 1e8; %% summable condition involved in stopping criteria of inner iteration
paraAlgo.In.SummableDegree = 1.01; %% degree for the series in the summable condition 

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
fprintf('PSNR: %f (before restoration)\n',psnr(Img_obs,Img_ref,paraModel.PoissonNoiseScale))
fprintf('PSNR: %f (after restoration)\n',History.PSNR(end))
