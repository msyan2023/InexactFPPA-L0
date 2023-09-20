function [DecFil, RecFil] = Generate_OneD_Filter(J)

JJ = 2*J + 1;

MatDCT = zeros(JJ,JJ);

for k=1:JJ
    
    for j=1:JJ
        
        if k==1
            delta = 1;
        else
            delta = sqrt(2);
        end
        
        MatDCT(k,j) = delta/JJ*cos((k-1)*(2*(j-1)+1)*pi/2/JJ);      %%  see equation (33) in the reference paper
        
    end
    
end

a = zeros(1,JJ);

DecMat_set = zeros(JJ,JJ,JJ);      %% set of decomposition matrix (decompose signal into framelet coefficients, that is, signal --> framelet coefficients)

for k = 1:JJ
    
    a(1:ceil(JJ/2)) = MatDCT(k,ceil(JJ/2):end);   %% see the equation below (35)
    
    DecMat_set(:,:,k) = TOEPLITZ(a,(-1)^(k-1)*a) + HANKEL((-1)^(k-1)*a, a);  %% see equation (35)
    
end

RecMat_set = zeros(JJ,JJ,JJ);   %% set of reconstrucation matrix (reconstruct signal from framelet coefficients, that is, framelet coefficients --> signal)

for k=1:JJ
    RecMat_set(:,:,k) = DecMat_set(:,:,k).';   %% H.' (H transpose) set
end

% rule: every row vector stores filter
DecFil = zeros(JJ, JJ);  %% decomposition filter
RecFil = zeros(JJ,JJ);  %% reconstruction filter

for k=1:JJ
    DecFil(k,:) = DecMat_set(J+1,:,k);
    RecFil(k,:) = RecMat_set(J+1,:,k);
end
end
