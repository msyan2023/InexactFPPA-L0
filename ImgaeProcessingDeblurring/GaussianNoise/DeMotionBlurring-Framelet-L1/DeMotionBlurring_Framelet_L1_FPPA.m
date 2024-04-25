function [u2,v2, History] = DeMotionBlurring_Framelet_L1_FPPA(paraAlgo, paraModel, Img_obs, Img_ref)
%{
%  General model: 
%
%        minimize varphi(u) + psi(Bu)
%
%  SingleStep FPPA: 
%
%  u^{k+1} = prox_{1/rho*varphi} (u^{k} - 1/rho*B.'*v^{k})
%  v^{k+1} = 1/beta*(I - prox_{beta*psi})( beta*v^{k} + B*(2u^{k+1}-u^{k}))
%  
%  convergence condition: beta*rho > norm(B,2)^2
%}

% Solves the following tight framlet model via SingleStep FPPA:

%    minimize_u 1/2*|| Ku - x ||_2^2 + lambda || Du ||_1
%                   u is an image (matrix) and x is observed motion blurred noisy image
%                   K is motion blurring(symmetric padding) kernel matrix 
%                   D is tight framelet matrix for image(2d signal) 

% Specify functions varphi, psi and matrix B in FPPA:
    %
    % varphi(u) = 1/2*|| Ku - x ||_2^2  
    %
    %           (prox_{varphi} needs the inverse of a matrix which can not be obtained and there is no fast algorithm in this blurring case. )
    %           (we treat the evaluation of prox_{varphi} as a sub optimization problem and use FPPA to obtain an approximated version.)
    %           (inner iteration is hence involved.)
    %                                    
    % psi(u) = lambda*|| u ||_1
    %
    % matrix B = Tight framelet matrix D for image(2d signal)   
    %          B:   Img --> Tight framelet coefficients   (decomposition)
    %          B':  Tight framelet coefficients --> Img   (reconstruction)
    %          Img === variable u in FPPA
    %          Tight framelet coefficients === variable v in FPPA
%% model parameter
lambda = paraModel.lambda;
 
J = paraModel.Framelet.J; JJ = 2*J + 1; 
DecFil = paraModel.Framelet.DecFil;
RecFil = paraModel.Framelet.RecFil;
%% algorithm parameter
% Outer Iteration parameters
MaxIter = paraAlgo.Out.MaxIter; %% maximal outer loop iteration
RelStop = paraAlgo.Out.RelStop;  %% relative stopping error tolerance
rho = paraAlgo.Out.rho;
beta = paraAlgo.Out.beta;

%% initial value
[N1,N2] = size(Img_ref);
u1 = zeros(N1,N2);  
w1 = zeros(N1,N2);  %% variable in inner iteration 
v1 = cell(JJ,JJ); 

for i=1:JJ
    for j=1:JJ
        v1{i,j} = zeros(N1,N2);
    end
end

% FPPA Iteration
for k = 1:MaxIter
    % u update
    temp_u = u1-1/rho*RecTightFramelet(v1, J, RecFil);
    [u2, w2, InnerNum] = InnerIteration_FPPA_psi_B(1/rho, temp_u, u1, w1, paraAlgo, paraModel, Img_obs,k);
    
    % v update
    temp_v = DecTightFramelet(2*u2-u1, J, DecFil);
    temp_v = cellfun(@(x,y) beta*x + y, v1, temp_v,'UniformOutput',false);
    
    v2 = cellfun(@(x) 1/beta*(x-prox_abs(x,beta*lambda)), temp_v, 'UniformOutput',false);
    
    
    % recording 
    History.PSNR(k) = mypsnr(Img_ref,u2);
    History.TargetValue(k) = TargetFunction(u2,paraModel, Img_obs);
    History.InnerNum(k) = InnerNum;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%Here is a difference against the non-working file%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    History.u_RelDiff(k) = norm(u2-u1)/norm(u2);     %%  this is 2-matrix norm (largest eigenvalue) instead of frobinus norm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if rem(k,20)==0
        fprintf('Iter: %d, TargetValue: %f;  PSNR: %2.6f; RelDiff: %2.6f\n', k, History.TargetValue(k), History.PSNR(k), History.u_RelDiff(k))
    end
    
    
    % stopping criteria
    if History.u_RelDiff(k) <= RelStop
        break
    end
    
    % update
    u1 = u2;
    v1 = v2;
    
    w1 = w2;
end

    
    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*|| ||_1$
        y = sign(x).*max(abs(x)-tau,0);
    end

end

