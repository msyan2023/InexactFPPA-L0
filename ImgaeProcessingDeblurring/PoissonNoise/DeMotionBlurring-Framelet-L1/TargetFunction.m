function y = TargetFunction(u,paraModel, Img_obs)
% Target Function Value & Loss Value

% For Gaussian Noise:
%    1/2*|| Ku - y ||_2^2 + lambda || Du ||_1
%
% For Poisson Noise:
%    <Ku,1> - <log(Ku),y> + lambda || Du ||_1
% 
% 
%   where  K is blurring kernel matrix 
%
%               D is DCT-induced tight framelet decomposition matrix   
%
%               y is Img_obs (given blurred and noisy image)
%
%               u --- image

 
lambda = paraModel.lambda;  % lambda: regularization parameter
J = paraModel.Framelet.J;  
DecFil = paraModel.Framelet.DecFil;
blur_filter = paraModel.Blur.blur_filter;

blur_u = imfilter(u, blur_filter,'symmetric');

Du = DecTightFramelet(u, J, DecFil);

% For Gaussian noise:
%  y = 1/2*norm(blur_u - Img_obs,'fro')^2 + lambda*sum(sum(abs([Du{:}])));  % Function value

% For Poisson noise:
y = sum(blur_u, "all") - sum(log(blur_u).*Img_obs,"all") + lambda*sum(sum(abs([Du{:}])));  % Function value

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

end











