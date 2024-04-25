function [u1, v1, History] = Classification_SquareHingeLoss_L1_FPPA(KernelMat, PredMat, LabelTrain, LabelTest, paraFPPA, paraModel)
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

%{ Solves the following tight framlet model via SingleStep FPPA:
%    minimize_u     sqHinge((SignMat.*K)*u)  + lambda || u ||_1
%                   u is coefficient for kernel function 
%                   SignMat is the sign matrix of training samples (diagnol matrix)
%                   K is Gaussian kernel matrix 
%                   sqHinge is square hinge loss              sqHinge(w) = 1/2*(max(1-w,0))^2    (differentiable)
%}


%{ Specify functions varphi, psi and matrix B in FPPA:
    %
    % varphi(u) = lambda*|| u ||_1 
    %                                    
    % psi(u) = square hinge loss 
    %
    % matrix B = (Sign Matrix) .* (The kernel matrix K)
%}
    
global MatB

lambda = paraModel.lambda;

rho = paraFPPA.rho;
beta = paraFPPA.beta;

MaxIter = paraFPPA.MaxIter;
RelStop = paraFPPA.RelStop;

NumShow = paraFPPA.NumShow;   %% number of showing progress


n=size(MatB,2); 

u1 = zeros(n,1);
v1 = zeros(n,1);

%   FPPA Iteration
for k = 1:1:MaxIter
    % u update
    u2 = prox_abs(u1-1/rho*MatB.'*v1, lambda/rho);
    
    temp = beta*v1 + MatB*(2*u2-u1);
    % v update
    v2 = 1/beta*(temp-prox_sqHinge(temp,beta));
    
    
    % recording 
    History.TargetValue(k) = lambda*norm(u2,1) + sum(sqHinge(MatB*u2));  % Function value
    History.NNZ(k) = nnz(u2);
    
    History.TrainAccuracy(k) = ShowAccuracyClassification(sign(KernelMat*u2), LabelTrain);
    History.TestAccuracy(k) =  ShowAccuracyClassification(sign(PredMat*u2), LabelTest);
    
    History.u_RelDiff(k) = norm(u2-u1)/norm(u2);
    History.v_RelDiff(k) = norm(v2-v1)/norm(v2);
    
    if rem(k,NumShow)==0
        fprintf('Iter: %d, TargetValue: %f; NNZ: %d; TRain: %0.2f%%; TEst: %0.2f%%; u_RelDiff: %0.6f; v_RelDiff: %0.6f\n',...
            k, History.TargetValue(k), History.NNZ(k), History.TrainAccuracy(k), History.TestAccuracy(k), History.u_RelDiff(k), History.v_RelDiff(k))
    end
    
    
    % stopping criteria 
    if History.u_RelDiff(k) < RelStop
        break
    end
    
    % update
    u1 = u2;
    v1 = v2;
end


    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||x||_1$
        y = sign(x).*max(abs(x)-tau,0);
    end

    function y = prox_sqHinge(x,beta)
        %% prox operator(gradient) for square hinge loss function with a constant rho:   beta*sqHinge  @   (x)
        z = (x+beta)/(1+beta);
        y = x.*(z>1) + z.*(z<=1);
    end

    function y = sqHinge(x)
        %% square hinge loss function 
        y = 1/2*(max(1-x,0)).^2;
    end
end