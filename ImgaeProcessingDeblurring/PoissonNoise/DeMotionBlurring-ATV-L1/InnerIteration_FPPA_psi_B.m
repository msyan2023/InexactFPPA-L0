function [u1, w1, InnerNum] = InnerIteration_FPPA_psi_B(const, EVALUATE_HERE, uLast, wLast, paraAlgo, paraModel, Img_obs,OutIterIndex)

% Inner Iteration solves the sub minimization problem  
%   prox_{const*(psi \circ K)} (EVALUATE_HERE)  = argmin{ const*psi(Ku) + 1/2*||u-EVALUATE_HERE||^2 : u}

%   "Img_obs" is blurred and noisy image and "K" is blurring kernel matrix
% 
%    For Gaussian Noise:  psi(Ku) = 1/2*||Ku-Img_obs||^2 
%    For Poisson Noise:   psi(Ku) = <Ku,1> - <log(Ku),Img_obs>


  % Inner Iteration FPPA:
  % u^{k+1} = prox_{1/rho*varphi}(  u^{k} - 1/rho*K.'*w^{k}  )         where varphi @ (u) = 1/2*||u-EVALUATE_HERE||^2
  % w^{k+1} = 1/beta*(I-prox_{beta*psi})( beta*w^{k} + K*(2*u^{k+1}-u^{k}))   
                            % where psi @ (w) = 1/2*||w-Img_obs||^2 for Gaussian Noise 
                            % where psi @ (w) = <w,1> - <log(w),Img_obs> for Poisson Noise
  
  % convergence condition: beta*rho>||K||_2^2

rho = paraAlgo.In.rho;
beta = paraAlgo.In.beta;
NormInnerSTOP = paraAlgo.In.NormInnerSTOP;
SummableDegree = paraAlgo.In.SummableDegree;

blur_filter = paraModel.Blur.blur_filter;

u1 = uLast;
w1 = wLast;

%% for Gaussian noise
% norm_diff = norm(const*MotionBlurringMat_Transp(grad_square(imfilter(u1,blur_filter,'symmetric'),Img_obs), blur_filter) + u1 - EVALUATE_HERE,'fro');

%% for Poisson noise
norm_diff = norm(const*MotionBlurringMat_Transp(grad_KL(imfilter(u1,blur_filter,'symmetric'),Img_obs), blur_filter) + u1 - EVALUATE_HERE,'fro');

InnerNum = 0;   %% record number of inner iterations

while (norm_diff>NormInnerSTOP/OutIterIndex^SummableDegree)||(InnerNum==0)
    % u-update
    temp_u = u1 - MotionBlurringMat_Transp(w1, blur_filter)/rho;         
    u2 = prox_square(temp_u, EVALUATE_HERE, 1/rho);
    
    % v-update
    temp_w = beta*w1 + imfilter(2*u2-u1,blur_filter,'symmetric');

    % for Gaussian Noise
    % w2 = (       temp_w      -     prox_square(temp_w, Img_obs, beta*const)       )/beta;
    
    % for Poisson Noise
    w2 = (       temp_w      -     prox_KL(temp_w, Img_obs, beta*const)       )/beta;
    
    % update
    u1 = u2;
    w1 = w2;
    InnerNum = InnerNum + 1;
    
    %% used for summable condition
    % for Gaussian noise
    % norm_diff = norm(const*MotionBlurringMat_Transp(grad_square(imfilter(u1,blur_filter,'symmetric'),Img_obs), blur_filter) + u1 - EVALUATE_HERE,'fro');
    
    % for Poisson noise
    norm_diff = norm(const*MotionBlurringMat_Transp(grad_KL(imfilter(u1,blur_filter,'symmetric'),Img_obs), blur_filter) + u1 - EVALUATE_HERE,'fro');
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function y = prox_square(u,x,tau)
        %% prox_{tau*1/2*||\cdot-x||_2^2} @ (u)
        y = (tau*x+u)/(tau+1);
    end

    function z = prox_KL(x, y, beta)
        %% prox_{beta*psi} @ (x)
         % where psi(\cdot) = <\cdot,1> - <log(\cdot),y>
         % where y stands for Img_obs, the noisy and blurred image 
         % This proximity operator is multi-valued 
         % z = ((x - beta) "+ or -" sqrt((beta-x).^2+4*y*beta))/2
         % we take the larger one, which is "+"
        z = ((x - beta) + sqrt((beta-x).^2+4*y*beta))/2;

    end

    function z = grad_square(x,y)
        %% gradient of square function f(x)=1/2*||x-y||^2
        % used in computing norm_diff for Gaussian noise
        z = x - y;
    end

    function z = grad_KL(x,y)
        %% gradient of KL function f(x)=<x,1> - <ln(x),y>
        z = 1 - y./x;
    end


end