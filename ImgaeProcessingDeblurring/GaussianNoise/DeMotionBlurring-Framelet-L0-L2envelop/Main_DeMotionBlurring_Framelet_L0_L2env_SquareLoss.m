

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%
%   (L0 norm regularization)                   L2 envelop
%
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%----------    FPPA solves the following model   (y is given blurred and noisy image)
%
%    min lambda*|u|_0 + lambda/gamma*1/2*||u-Dv||_2^2 + psi(Bv,y)
%
%    where  psi(Bv,y) = 1/2*||Bv-y||_2^2
%
%               B is blurring kernel matrix (motion blurring with symmetric(reflexive/mirror) padding condition)
%                     the computation of prox_{psi\circ B} needs an inverse of a matrix and for this case of blurring,
%                     there is no fast algorthm(not like padding with periodic condition) provided, thus inner iteration
%                     for estimating prox_{psi\circ B} is involved
%
%               D is DCT-induced tight framelet decomposition matrix
%
%               u --- framelet coefficient    v --- image

clear, clc, close all
format short
dbstop if error

restoredefaultpath
addpath('data')
addpath('GenerateTightFrameletFilter')
addpath('TightFramelet')
addpath('Motion-blurring')

disp('Deblurring-Square Loss-L0 Regularization-L2 envelop-FPPA')
disp('Stopping Criteria of Inner Iteration: Sufficient Decreasing Condition && Sumable condition (Alternating Minimization Algorithm)')
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
paraModel.lambda = 0.2;   %% regularization parameter
paraModel.gamma = 2;  %%  envelop parameter

%%%%%%%%%%%%%%%%%   Use initial value = 0 %%%%%%%%%%%%%%%%%%

%----- Outer iteration parameter
paraAlgo.Out.MaxIter = 5000;  %% maximal iteration for Outer iteration
paraAlgo.Out.alpha = 0.99;  %% average parameter  (u2 = prox{ (1-alpha)*u1 + alpha*D*v1 }    alpha \in (0,1)
paraAlgo.Out.theta = 1;     %% average parameter  (v2 = prox{ theta*D.'*u2 + (1-theta)*v1 }    theta \in (0,1]

paraAlgo.Out.RelStop = 1e-5;

%----- Inner iteration parameter
% paraAlgo.In.rho --> parameter p in the paper
% paraAlgo.In.beta --> parameter q in the paper
paraAlgo.In.rho = 0.1;   
paraAlgo.In.beta = (1+1e-6) * 4/paraAlgo.In.rho;   %% convergence condition: beta*rho>||blurring kernel matrix||_2^2
paraAlgo.In.FacInnerSTOP = 0.99 * 1000;  %%   sufficient decreasing condition involved in stopping criteria of inner iteration
paraAlgo.In.NormInnerSTOP = 1e6; %% summable condition involved in stopping criteria of inner iteration

paraModel
paraAlgo
paraAlgo.Out
paraAlgo.In
fprintf('Threshold of ell0 truncation = %f\n\n',sqrt(2*paraModel.gamma*paraAlgo.Out.alpha))
disp('-----------------------------------------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Training
tic
[u, v, w, History] = DeMotionBlurring_Framelet_L0_L2env_FPPA(paraAlgo, paraModel, Img_obs, Img_ref);
time = toc;
fname = sprintf('L0-DeMotionBlurring_PSNR%2.5f_lambda%2.2f_gamma%2.2f_rho%2.2f_alpha%0.2f.mat',...
    History.PSNR(end), paraModel.lambda, paraModel.gamma, paraAlgo.In.rho, paraAlgo.Out.alpha)
save(fname, 'u', 'v', 'w', 'History', 'paraModel', 'paraAlgo')


%% show result
figure, imshow(uint8(Img_ref)), title('Original Image')
figure, imshow(uint8(Img_blur)), title('Blurred Image')
figure, imshow(uint8(Img_obs)), title('Observed Image')
figure, imshow(uint8(v)), title('Restored Image')

figure, plot(History.PSNR), title('PSNR'), xlabel('Iteration')
figure, plot(History.TargetValue), title('TargetValue'), xlabel('Iteration')
figure, plot(History.InnerNum), title('InnerNum'), xlabel('Iteration')


fprintf('Time: %f s\n', time)
fprintf('PSNR: %f (before restoration)\n',mypsnr(Img_ref,Img_obs))
fprintf('PSNR: %f (after restoration)\n',History.PSNR(end))
