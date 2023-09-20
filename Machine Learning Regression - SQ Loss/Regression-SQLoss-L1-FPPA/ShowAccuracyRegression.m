function MSE = ShowAccuracyRegression(pred, Labels)

NumPoints = length(Labels); 

MSE = sum((pred - Labels).^2)/NumPoints;   %% mean square error

end
