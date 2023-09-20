function [u1, w1, InnerNum] = InnerIteration_FPPA_psi_B(const, EVALUATE_HERE, uLast, wLast, paraAlgo, paraModel, Img_obs,OutIterIndex)

% Inner Iteration solves the sub minimization problem  
%   prox_{const*(psi \circ K)} (EVALUATE_HERE)  = argmin{ const*psi(Ku) + 1/2*||u-EVALUATE_HERE||^2 : u}

%                       (---psi(Ku) = 1/2*||Ku-Img_obs||^2 where "Img_obs" is blurred and noisy image and K is blurring kernel matrix---)

  % Inner Iteration FPPA:
  % u^{k+1} = prox_{1/rho*varphi}(  u^{k} - 1/rho*K.'*w^{k}  )         where varphi @ (u) = 1/2*||u-EVALUATE_HERE||^2
  % w^{k+1} = 1/beta*(I-prox_{beta*psi})( beta*w^{k} + K*(2*u^{k+1}-u^{k}))   where psi @ (w) = const*1/2*||w-Img_obs||^2
  
  % convergence condition: beta*rho>||K||_2^2

rho = paraAlgo.In.rho;
beta = paraAlgo.In.beta;
NormInnerSTOP = paraAlgo.In.NormInnerSTOP;

blur_filter = paraModel.Blur.blur_filter;

u1 = uLast;
w1 = wLast;

norm_diff = norm(EVALUATE_HERE - u1 - const*MotionBlurringMat_Transp(imfilter(u1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
InnerNum = 0;   %% record number of inner iterations

while (norm_diff>NormInnerSTOP/OutIterIndex^2)||(InnerNum==0)
    % u-update
    temp_u = u1 - MotionBlurringMat_Transp(w1, blur_filter)/rho;         
    u2 = prox_square(temp_u, EVALUATE_HERE, 1/rho);
    
    % v-update
    temp_w = beta*w1 + imfilter(2*u2-u1,blur_filter,'symmetric');
    w2 = (       temp_w      -     prox_square(temp_w, Img_obs, beta*const)       )/beta;
    
    
    % update
    u1 = u2;
    w1 = w2;
    InnerNum = InnerNum + 1;
    
    % used for summable condition
    norm_diff = norm(EVALUATE_HERE - u1 - const*MotionBlurringMat_Transp(imfilter(u1,blur_filter,'symmetric')-Img_obs, blur_filter), 'fro');
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function y = prox_square(u,x,tau)
        %% prox_{tau*1/2*||\cdot-x||_2^2} @ (u)
        y = (tau*x+u)/(tau+1);
    end


end