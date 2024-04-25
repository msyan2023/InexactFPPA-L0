function Accuracy = ShowAccuracyClassification(pred, Labels)

NumPoints = length(Labels); 

Accuracy = sum((pred - Labels)==0)/NumPoints*100;


end
