function y = TargetFunction(u,paraModel, Img_obs)
% Target Function Value & Loss Value

%    1/2*|| Ku - y ||_2^2 + lambda || Du ||_1
%
%    where  K is blurring kernel matrix 
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

 y = 1/2*norm(blur_u - Img_obs,'fro')^2 + lambda*sum(sum(abs([Du{:}])));  % Function value


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%


end