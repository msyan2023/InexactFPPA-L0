function [u1, v1, w1, History,k] = Alternating_SQLoss_L0_L2env(KernelMat, PredMat, LabelTrain, LabelTest, paraAlgo, paraModel)
%%  FPPA solves the model
%    min lambda*||u||_0 + lambda/gamma*1/2*||u-v||_2^2 + psi(Bv)
%    where psi(x) = 1/2*||x-LabelTrain||^2    square loss
%                B = KernelMat

%    u:  no bias  (coefficient before kernel function K(x,cdot))
%    Outer Iteration:  u^{k+1} = prox_{alpha*gamma*|| ||_0}  @ ( alpha*v^{k} + (1-alpha)*u^{k} )
%                              v^{k+1} = prox_{theta*gamma/lambda*(psi \circ B)} @ ( theta*u^{k+1} + (1-theta)*v^{k} )     ----->   Inner Iteration is needed here


global MatB

%% model parameter
lambda = paraModel.lambda;   %% regularization parameter
gamma = paraModel.gamma;   %% envelop parameter: L0 envelop converges to L0 when gamma goes to 0

%% algorithm parameter
% Outer Iteration parameters
MaxIter = paraAlgo.Out.MaxIter; %% maximal outer loop iteration
alpha = paraAlgo.Out.alpha; %% average parameter  (u2 = prox{ (1-alpha)*u1 + alpha*v1) }    alpha \in (0,1)
theta = paraAlgo.Out.theta; %% average parameter  (v2 = prox{ theta*u2 + (1-theta)*v1) }   theta \in (0,1]
RelStop = paraAlgo.Out.RelStop;  %% relative stopping error tolerance
NumShow = paraAlgo.Out.NumShow;   %% number of showing progress

FacInnerSTOP = paraAlgo.In.FacInnerSTOP;  %% factor of stopping criteria for inner iteration
InnerSTOP = FacInnerSTOP*lambda*(1/alpha-1)/gamma;  %% This is "rho prime" in the paper
%% initial value
n=size(MatB,2);

u1 = zeros(n,1);
v1 = zeros(n,1);
w1 = zeros(n,1);


%%   Outer Iteration
for k = 1:1:MaxIter
    % u-update
    u2 = prox_ell0((1-alpha)*u1+alpha*v1, alpha*gamma);
    
    % v-update (inner iteration)
    InnerTOL = InnerSTOP * norm(u2-u1)^2/2 + TargetFunction(u2,v1,paraModel,LabelTrain);
    [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(theta*gamma/lambda, theta*u2+(1-theta)*v1, LabelTrain, u2, v1, w1, paraAlgo, paraModel, InnerTOL, k);
%     [v2, w2, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B_1stepInnerIteration(theta*gamma/lambda, theta*u2+(1-theta)*v1, LabelTrain, u2, v1, w1, paraAlgo, paraModel, InnerTOL, k);
    %% Record
    History.TargetValue(k) = TargetValue; % Function value
    History.InnerNum(k) = InnerNum; % Number of inner iterations
    History.NNZ(k) = nnz(u2);
    History.TrainMSE(k) = ShowAccuracyRegression(KernelMat*u2, LabelTrain);      
    History.TestMSE(k) =  ShowAccuracyRegression(PredMat*u2, LabelTest);
   
    History.u_RelDiff(k) = norm(u2-u1)/norm(u2);
    History.v_RelDiff(k) = norm(v2-v1)/norm(v2);
    %% Show Progress
    if rem(k,NumShow)==0
        fprintf('Iter: %d, TargetValue: %f; NNZ: %d; TRain: %0.5f; TEst: %0.5f; u_RelDiff: %0.6f; v_RelDiff: %0.6f; InnerIteration:%d\n',...
            k, History.TargetValue(k), History.NNZ(k), History.TrainMSE(k), History.TestMSE(k), History.u_RelDiff(k), History.v_RelDiff(k), sum(History.InnerNum(end-NumShow+1:end)))
    end
    
    %% stopping criteria
    if History.u_RelDiff(k)<RelStop
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