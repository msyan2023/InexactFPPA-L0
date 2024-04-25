function [u1, v1, History,k] = Regression_SQLoss_L1_FPPA(KernelMat, PredMat, LabelTrain, LabelTest, paraFPPA, paraModel)
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
%    minimize_u     SQ(K*u)  + lambda || u ||_1
%                   u is coefficient for kernel function 
%                   K is Gaussian kernel matrix 
%                   SQ is square loss              SQ(w) = 1/2*||w-LabelTrain||^2    (differentiable)
%}

%{ Specify functions varphi, psi and matrix B in FPPA:
    %
    % varphi(u) = lambda*|| u ||_1 
    %                                    
    % psi(u) = SQ  
    %
    % matrix B = The kernel matrix K
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
    v2 = 1/beta*(temp-prox_square(temp,beta,LabelTrain));
    
    
    % recording 
    History.TargetValue(k) = lambda*norm(u2,1) + 1/2*sum((KernelMat*u2-LabelTrain).^2);  % Function value
    History.NNZ(k) = nnz(u2);
    
    History.TrainMSE(k) = ShowAccuracyRegression(KernelMat*u2, LabelTrain);
    History.TestMSE(k) =  ShowAccuracyRegression(PredMat*u2, LabelTest);
    
    History.u_RelDiff(k) = norm(u2-u1)/norm(u2);
    History.v_RelDiff(k) = norm(v2-v1)/norm(v2);
    %% Show Progress
    if rem(k,NumShow)==0
        fprintf('Iter: %d, TargetValue: %f; NNZ: %d; TRain: %0.5f; TEst: %0.5f; u_RelDiff: %0.6f; v_RelDiff: %0.6f\n',...
            k, History.TargetValue(k), History.NNZ(k), History.TrainMSE(k), History.TestMSE(k), History.u_RelDiff(k), History.v_RelDiff(k))
    end
    
    
    % stopping criteria 
    if History.u_RelDiff(k) < RelStop
        break
    end

%     if (k>20000) && (History.TrainAccuracy(k)<50)     %% after 20000 iterations, if the training accuracy is lower than the 50%, it is mostly hopeless to get good result so we quit
%         disp('QUIT because the training accuracy is lower than 50% after 20000 iterations')
%         break
%     end

    
    % update
    u1 = u2;
    v1 = v2;
end


    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||x||_1$
        y = sign(x).*max(abs(x)-tau,0);
    end

    function y = prox_square(x,beta,LabelTrain)
        %% prox operator(gradient) for square loss function with a constant rho:   beta*1/2*||  - LabelTrain ||^2  @  (x)
        y = (beta*LabelTrain + x)/(beta+1);
    end
 
end