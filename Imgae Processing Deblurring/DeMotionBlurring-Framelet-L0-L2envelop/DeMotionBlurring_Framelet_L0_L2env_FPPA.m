function [u1, v1, w1, History] = DeMotionBlurring_Framelet_L0_L2env_FPPA(paraAlgo, paraModel, Img_obs, Img_ref)

%{
----------    FPPA solves the following model   (y is given blurred and noisy image)

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
%               u --- framelet coefficient (cell structure)   v --- image
%}

%   Outer Iteration:  u^{k+1} = prox_{alpha*gamma*|| ||_0}  @ ( alpha*D*v^{k} + (1-alpha)*u^{k} )
%                             v^{k+1} = prox_{theta*gamma/lambda*(psi \circ B)} @ (theta*D_transpose*u^{k+1} + (1-theta)*v^{k})     ----->   Inner Iteration is needed here


%% model parameter
lambda = paraModel.lambda;   %% regularization parameter
gamma = paraModel.gamma;   %% envelop parameter: L0 envelop converges to L0 when gamma goes to 0

J = paraModel.Framelet.J; JJ = 2*J+1; 
DecFil = paraModel.Framelet.DecFil;
RecFil = paraModel.Framelet.RecFil;
%% algorithm parameter
% Outer Iteration parameters
MaxIter = paraAlgo.Out.MaxIter; %% maximal outer loop iteration
alpha = paraAlgo.Out.alpha;  %% average parameter  (u2 = prox{ (1-alpha)*u1 + alpha*D*v1 }    alpha \in (0,1)
theta = paraAlgo.Out.theta;   %% average parameter  (v2 = prox{ theta*D.'*u2 + (1-theta)*v1 }    theta \in (0,1]

RelStop = paraAlgo.Out.RelStop;  %% relative stopping error tolerance

FacInnerSTOP = paraAlgo.In.FacInnerSTOP;  %% factor of stopping criteria for inner iteration
InnerSTOP = FacInnerSTOP*lambda*(1/alpha-1)/gamma;  %% This is "rho prime" in the paper  (in fact in the paper rho=lambda*(1/alpha-1)/gamma, rho'=InnerSTOP*rho is some number less then rho)
%% initial value
[N1,N2] = size(Img_obs);
u1 = cell(JJ,JJ);
for i=1:JJ
    for j=1:JJ
        u1{i,j} = zeros(N1,N2);
    end
end
v1 = zeros(N1,N2);
w1 = zeros(N1,N2);


%%   Outer Iteration
for k = 1:1:MaxIter
    % u-update
    Dv1 = DecTightFramelet(v1, J, DecFil);
    temp_u = cellfun(@(x,y) alpha*x+(1-alpha)*y, Dv1, u1, 'UniformOutput', false);
    u2 = cellfun(@(x) prox_ell0(x, alpha*gamma), temp_u, 'UniformOutput', false);
    
    % v-update (inner iteration)
    u_diff = sum(sum(cellfun(@(x,y) norm(x-y,'fro')^2, u2,u1, 'UniformOutput', true)));
    InnerTOL = InnerSTOP/2 * u_diff + TargetFunction(u2,v1,Dv1,paraModel,Img_obs);
    
    temp_v = theta*RecTightFramelet(u2, J, RecFil) + (1-theta)*v1;
    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(theta*gamma/lambda, temp_v, u2, v1, w1, paraAlgo, paraModel, Img_obs, InnerTOL, k);
%     [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B_1stepInnerIteration(theta*gamma/lambda, temp_v, u2, v1, w1, paraAlgo, paraModel, Img_obs, InnerTOL, k);
    %% Record
    History.TargetValue(k) = TargetValue; % Function value
    History.InnerNum(k) = InnerNum; % Number of inner iterations
    History.NNZ(k) = sum(sum(cellfun(@(x) nnz(x), u2, 'UniformOutput', true)));  
    History.PSNR(k) = mypsnr(Img_ref, v2);
 
    History.u_RelDiff(k) = sqrt(u_diff/sum(sum(cellfun(@(x) norm(x,'fro')^2, u2, 'UniformOutput', true))));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%Here is a difference against the non-working file%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    History.v_RelDiff(k) = norm(v2-v1)/norm(v2);     %%  this is 2-matrix norm (largest eigenvalue) instead of frobinus norm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Show Progress
    if rem(k,20)==0
        fprintf('Iter: %d, TargetValue: %f,    nnz: %d,    PSNR: %2.6f,   RelDiff: %f\n', k, History.TargetValue(k), History.NNZ(k), History.PSNR(k), History.v_RelDiff(k))
        if rem(k,1000)==0
            fprintf('\n')
        end
    end
    
    if History.v_RelDiff(k)<RelStop
        break
    end
    %% Update
    u1 = u2;
    v1 = v2;
    w1 = w2;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function x = prox_ell0(x,tau)
        %%  prox_{tau*|| ||_0} @ (x)
        x(abs(x)<sqrt(2*tau)) = 0;
    end


end