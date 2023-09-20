function y = TargetFunction(u,v,Dv,paraModel, Img_obs)
% Target Function Value & Loss Value

%    lambda*|u|_0 + lambda/gamma*1/2*||u-Dv||_2^2 + psi(Bv,y)
%
%    where  psi(Bv,y) = 1/2*||Bv-y||_2^2    and     y==Img_obs (blurred and noisy image)

%               u --- framelet coefficient (cell structure)    v --- image

 
lambda = paraModel.lambda;  % lambda: regularization parameter
gamma = paraModel.gamma;  % gamma: envelop parameter

blur_filter = paraModel.Blur.blur_filter;

blur_v = imfilter(v, blur_filter,'symmetric');

coeff_diff = sum(sum(cellfun(@(x,y) norm(x-y,'fro')^2, u, Dv, 'UniformOutput',true)));

nnz_u = sum(sum(cellfun(@(x) nnz(x), u, 'UniformOutput',true)));

 y = lambda*nnz_u + lambda/gamma/2*coeff_diff + 1/2*norm(blur_v-Img_obs,'fro')^2;  % Function value


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%


end