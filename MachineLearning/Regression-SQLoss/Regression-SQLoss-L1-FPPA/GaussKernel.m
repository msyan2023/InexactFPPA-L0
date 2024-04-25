function [GaussDistanceMat, DistanceMat] = GaussKernel(TrainPoints,TestPoints, sigma)

normTrain = sum(TrainPoints.*TrainPoints,1); %% matrix of norm^2 of each point
normTest = sum(TestPoints.*TestPoints,1); %% matrix of norm^2 of each point

[GridnormTrain, GridnormTest] = meshgrid(normTrain,normTest);
DistanceMat = GridnormTrain+GridnormTest-2*TestPoints'*TrainPoints;  %% matrix of ||x-y||^2
GaussDistanceMat = exp(-DistanceMat/(2*sigma^2));

assert (size(GaussDistanceMat,1) == size(TestPoints,2))

assert (size(GaussDistanceMat,2) == size(TrainPoints,2))

end