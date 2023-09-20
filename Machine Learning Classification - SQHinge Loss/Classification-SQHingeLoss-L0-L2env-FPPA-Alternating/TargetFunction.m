function y = TargetFunction(u,v,paraModel)
% Target Function Value & Loss Value

global MatB
lambda = paraModel.lambda;  % lambda: regularization parameter
gamma = paraModel.gamma;  % gamma: envelop parameter

LossValue = sum(sqHinge(MatB*v));
y = lambda*nnz(u) + lambda/gamma*norm(u-v,2)^2/2 + LossValue;  % Function value


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%


end