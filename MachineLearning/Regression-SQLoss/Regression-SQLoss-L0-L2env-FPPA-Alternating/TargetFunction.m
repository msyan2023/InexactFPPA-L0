function y = TargetFunction(u,v,paraModel, LabelTrain)
% Target Function Value & Loss Value

global MatB
lambda = paraModel.lambda;  % lambda: regularization parameter
gamma = paraModel.gamma;  % gamma: envelop parameter

LossValue = 1/2*sum((MatB*v-LabelTrain).^2);
y = lambda*nnz(u) + lambda/gamma*norm(u-v,2)^2/2 + LossValue;  % Function value


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%


end