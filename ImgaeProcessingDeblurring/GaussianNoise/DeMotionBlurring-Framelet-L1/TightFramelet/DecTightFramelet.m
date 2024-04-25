function FrameletCoeff = DecTightFramelet(Img, J, DecFil)
%% Decomposition: Image --> Tight Framelet Coefficients
% DecFil: matrix whose row stores 1 dimensional decomposition filters (k-th row <--> k-th decomposition filter)
JJ = 2*J + 1;
FrameletCoeff = cell(JJ,JJ);
for i = 1:JJ
    for j = 1:JJ
        FrameletCoeff{i,j} = imfilter(Img, DecFil(j,:)'*DecFil(i,:), 'symmetric');    
        % for column of image, use j-th decomposition filter
        % for row of image, use i-th decomposition filter
    end
end
end
