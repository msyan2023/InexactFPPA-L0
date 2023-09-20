function [v1, w1, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(const, EVALUATE_HERE, uLast, vLast, wLast, paraAlgo, paraModel, Img_obs, InnerTOL,OutIterIndex)

% Inner Iteration solves the sub minimization problem
%   prox_{const*(psi \circ K)} (EVALUATE_HERE)  = argmin{ const*psi(Kv) + 1/2*||v-EVALUATE_HERE||^2 : v}   (the standard function corresponding to proximity operator)
%                                                                      = argmin{ 1/2*||Kv-Img_obs||^2 + 1/2*||v-EVALUATE_HERE||^2/const : v}   
%                       (---psi(Kv) = 1/2*||Kv-Img_obs||^2 where "Img_obs" is blurred and noisy image and K is blurring kernel matrix---)

  % Inner Iteration FPPA:
  % v^{k+1} = prox_{1/rho*varphi}(  v^{k} - 1/rho*K.'*w^{k}  )         where varphi @ (v) = 1/const*1/2*||v-EVALUATE_HERE||^2
  % w^{k+1} = 1/beta*(I-prox_{beta*psi})( beta*w^{k} + K*(2*v^{k+1}-v^{k}))   where psi @ (w) = 1/2*||w-Img_obs||^2
  
  % convergence condition: beta*rho>||K||_2^2


rho = paraAlgo.In.rho;
beta = paraAlgo.In.beta;
NormInnerSTOP = paraAlgo.In.NormInnerSTOP;

blur_filter = paraModel.Blur.blur_filter;

J = paraModel.Framelet.J;
DecFil = paraModel.Framelet.DecFil;

v1 = vLast;
w1 = wLast;
Dv1 = DecTightFramelet(v1, J, DecFil);

TargetValue = TargetFunction(uLast,v1,Dv1,paraModel,Img_obs); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%Here is a difference against the non-working file%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norm_diff = 1/const*norm(EVALUATE_HERE - v1 - const*MotionBlurringMat_Transp(imfilter(v1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
% norm_diff = norm(EVALUATE_HERE - v1 - const*MotionBlurringMat_Transp(imfilter(v1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

InnerNum = 0;   %% record number of inner iterations

while ((TargetValue>InnerTOL)||(norm_diff>NormInnerSTOP/OutIterIndex^2))||(InnerNum==0)
    % v-update
    temp_v = v1 - MotionBlurringMat_Transp(w1, blur_filter)/rho;         
    v2 = prox_square(temp_v, EVALUATE_HERE, 1/const/rho);
    
    % w-update
    temp_w = beta*w1 + imfilter(2*v2-v1,blur_filter,'symmetric');
    w2 = (       temp_w      -     prox_square(temp_w, Img_obs, beta)       )/beta;
    
    
    % update
    v1 = v2;
    w1 = w2;
    InnerNum = InnerNum + 1;
    
    %  used for sufficient decreasing condition 
    Dv1 = DecTightFramelet(v1, J, DecFil);
    TargetValue = TargetFunction(uLast,v1,Dv1,paraModel,Img_obs);   % Compute current function @ (uLast, v1)
    
    % used for summable condition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%Here is a difference against the non-working file%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    norm_diff = 1/const*norm(EVALUATE_HERE - v1 - const*MotionBlurringMat_Transp(imfilter(v1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
    % norm_diff = norm(EVALUATE_HERE - v1 - const*MotionBlurringMat_Transp(imfilter(v1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function y = prox_square(u,x,tau)
        %% prox_{tau*1/2*||\cdot-x||_2^2} @ (u)
        y = (tau*x+u)/(tau+1);
    end


end