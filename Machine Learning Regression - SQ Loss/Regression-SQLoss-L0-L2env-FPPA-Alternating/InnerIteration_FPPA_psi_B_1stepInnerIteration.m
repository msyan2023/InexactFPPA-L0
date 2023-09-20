function [v1, w1, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B_1stepInnerIteration(const, EVALUATE_HERE, LabelTrain, uLast, vLast, wLast, paraAlgo, paraModel, InnerTOL, OutIterIndex)

% Inner Iteration solves the sub minimization problem for some fixed x
%   prox_{const*(SQ \circ B)} (EVALUATE_HERE) = argmin{ const*SQ(Bv) + ||v-EVALUATE_HERE||^2/2 : v}
%                                             = argmin{ SQ(Bv) + 1/2*1/const*||v-EVALUATE_HERE||^2 : v}

%                                             SQ(u) = 1/2*||u-LabelTrain||^2         

  % Inner Iteration FPPA:
  % v^{k+1} = prox_{1/rho*varphi}(  v^{k} - 1/rho*B.'*w^{k}  )         where varphi @ (x) = 1/const*1/2*||v-EVALUATE_HERE||^2
  % w^{k+1} = 1/beta*(I-prox_{beta*psi})( beta*w^{k} + B*(2*v^{k+1}-v^{k}))     where psi @ (x) = 1/2*||u-LabelTrain||^2    ------>  square loss 
  
  % convergence condition: beta*rho>||B||_2^2

global MatB

rho = paraAlgo.In.rho;
beta = paraAlgo.In.beta;
NormInnerSTOP = paraAlgo.In.NormInnerSTOP;

v1 = vLast;
w1 = wLast;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   sufficient decreasing condition:  %%%%%%%%%%%%%%%%%%%%
TargetValue = TargetFunction(uLast,v1,paraModel,LabelTrain);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   summable condition:  %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%     || gradient of "SQ(Bv) + 1/2*1/const*||v-EVALUATE_HERE||^2" @ (v1) ||_2 <= C/k^2
norm_diff = norm(MatB.'*(MatB*v1-LabelTrain)+(v1-EVALUATE_HERE)/const , 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


InnerNum = 0;   %% record number of inner iterations

% while ((TargetValue>InnerTOL)||(norm_diff>NormInnerSTOP/OutIterIndex^2))||(InnerNum==0)
    % v-update
    temp_v = v1-MatB.'*w1/rho;
    v2 = prox_square(temp_v, EVALUATE_HERE, 1/const/rho);
    
    % w-update
    temp_w = beta*w1 + MatB*(2*v2-v1);
    w2 = (       temp_w      -     prox_square(temp_w,LabelTrain, beta)       )/beta;
    
    % update
    v1 = v2;
    w1 = w2;
    InnerNum = InnerNum + 1; 
    
    % sufficient decreasing condition: Compute current function @ (uLast, v1)
    TargetValue = TargetFunction(uLast,v1,paraModel,LabelTrain);
   
    % summable condition 
    norm_diff = norm(MatB.'*(MatB*v1-LabelTrain)+(v1-EVALUATE_HERE)/const , 2);
    
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 

    function y = prox_square(u,x,tau)
        %% prox_{tau*1/2*||\cdot-x||_2^2} @ (u)
        y = (tau*x+u)/(tau+1);
    end

 


end