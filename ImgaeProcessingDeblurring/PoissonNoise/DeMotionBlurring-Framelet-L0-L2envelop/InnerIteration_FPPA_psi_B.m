function [v1, w1, InnerNum, TargetValue] = InnerIteration_FPPA_psi_B(const, EVALUATE_HERE, uLast, vLast, wLast, paraAlgo, paraModel, Img_obs, InnerTOL,OutIterIndex)

% Inner Iteration solves the sub minimization problem
%   prox_{const*(psi \circ K)} (EVALUATE_HERE)  = argmin{ const*psi(Kv) + 1/2*||v-EVALUATE_HERE||^2 : v}   (the standard function corresponding to proximity operator)
 
%   "Img_obs" is blurred and noisy image and "K" is blurring kernel matrix
% 
%    For Gaussian Noise:  psi(Kv) = 1/2*||Kv-Img_obs||^2 
%    For Poisson Noise:   psi(Kv) = <Kv,1> - <log(Kv),Img_obs>

  % Inner Iteration FPPA:
  % v^{k+1} = prox_{1/rho*varphi}(  v^{k} - 1/rho*K.'*w^{k}  )         where varphi @ (v) = 1/const*1/2*||v-EVALUATE_HERE||^2
  % w^{k+1} = 1/beta*(I-prox_{beta*psi})( beta*w^{k} + K*(2*v^{k+1}-v^{k}))
                            % where psi @ (w) = 1/2*||w-Img_obs||^2 for Gaussian Noise 
                            % where psi @ (w) = <w,1> - <log(w),Img_obs> for Poisson Noise 
  
  % convergence condition: beta*rho>||K||_2^2


rho = paraAlgo.In.rho;
beta = paraAlgo.In.beta;
NormInnerSTOP = paraAlgo.In.NormInnerSTOP;
SummableDegree = paraAlgo.In.SummableDegree;

blur_filter = paraModel.Blur.blur_filter;

J = paraModel.Framelet.J;
DecFil = paraModel.Framelet.DecFil;

v1 = vLast;
w1 = wLast;
Dv1 = DecTightFramelet(v1, J, DecFil);

TargetValue = TargetFunction(uLast,v1,Dv1,paraModel,Img_obs); 

%% for Gaussian noise
% norm_diff = norm(const*MotionBlurringMat_Transp(grad_square(imfilter(v1,blur_filter,'symmetric'),Img_obs), blur_filter) + v1 - EVALUATE_HERE,'fro');

%% for Poisson noise
norm_diff = norm(const*MotionBlurringMat_Transp(grad_KL(imfilter(v1,blur_filter,'symmetric'),Img_obs), blur_filter) + v1 - EVALUATE_HERE,'fro');


InnerNum = 0;   %% record number of inner iterations

while ((TargetValue>InnerTOL)||(norm_diff>NormInnerSTOP/OutIterIndex^SummableDegree))||(InnerNum==0)
    % v-update
    temp_v = v1 - MotionBlurringMat_Transp(w1, blur_filter)/rho;         
    v2 = prox_square(temp_v, EVALUATE_HERE, 1/const/rho);
    
    % w-update
    temp_w = beta*w1 + imfilter(2*v2-v1,blur_filter,'symmetric');

    % for Gaussian Noise 
    % w2 = (       temp_w      -     prox_square(temp_w, Img_obs, beta)       )/beta;
    
    % for Poisson Noise 
    w2 = (       temp_w      -     prox_KL(temp_w, Img_obs, beta)       )/beta;
    
    % update
    v1 = v2;
    w1 = w2;
    InnerNum = InnerNum + 1;
    
    %  used for sufficient decreasing condition 
    Dv1 = DecTightFramelet(v1, J, DecFil);
    TargetValue = TargetFunction(uLast,v1,Dv1,paraModel,Img_obs);   % Compute current function @ (uLast, v1)
    
    %% used for summable condition
    % for Gaussian noise
    % norm_diff = norm(const*MotionBlurringMat_Transp(grad_square(imfilter(v1,blur_filter,'symmetric'),Img_obs), blur_filter) + v1 - EVALUATE_HERE,'fro');
    
    % for Poisson noise
    norm_diff = norm(const*MotionBlurringMat_Transp(grad_KL(imfilter(v1,blur_filter,'symmetric'),Img_obs), blur_filter) + v1 - EVALUATE_HERE,'fro');
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function y = prox_square(u,x,tau)
        %% prox_{tau*psi} @ (u)
         % where psi(\cdot) = 1/2*||\cdot-x||_2^2
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