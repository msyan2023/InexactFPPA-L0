function y = TargetFunction(u,paraModel, Img_obs)
% Target Function Value & Loss Value

%    1/2*|| Ku - y ||_2^2 + lambda || Du ||_1
%
%    where  K is blurring kernel matrix 
%
%               D is first order difference matrix
%
%               y is Img_obs (given blurred and noisy image)
%
%               u --- image

 
lambda = paraModel.lambda;  % lambda: regularization parameter
 
blur_filter = paraModel.Blur.blur_filter;

blur_u = imfilter(u, blur_filter,'symmetric');

[diff1,diff2] = DiffMat(u);

 y = 1/2*norm(blur_u - Img_obs,'fro')^2 + lambda*(sum(sum(abs(diff1)+abs(diff2))));  % Function value


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%


end