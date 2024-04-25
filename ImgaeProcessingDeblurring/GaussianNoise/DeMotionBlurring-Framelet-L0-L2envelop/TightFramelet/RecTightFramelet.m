function Img = RecTightFramelet(FrameletCoeff, J, RecFil)
%% Reconstruction: Tight Framelet Coefficients --> Image
% RecFil: matrix whose row stores 1 dimensional reconstruction filters  (k-th row <--> k-th reconstruction filter)
JJ = 2*J + 1;
Img_piece_cell = cell(JJ,JJ);
for i = 1:JJ
    for j = 1:JJ
        
        if mod(j,2)==1
            temp = wextend('ar', 'sym',FrameletCoeff{i,j}, J);
        else
            temp = wextend('ar', 'asym',FrameletCoeff{i,j}, J);
        end
        
        if mod(i,2)==1
            temp = wextend('ac', 'sym',temp, J);
        else
            temp = wextend('ac', 'asym',temp, J);
        end
        
        Img_piece = imfilter(temp, RecFil(j,:)'*RecFil(i,:));
        % for column of image, use j-th decomposition filter
        % for row of image, use i-th decomposition filter
        
        Img_piece_cell{i,j} = Img_piece(J+1:end-J, J+1:end-J);   %% omit the 0 padding part while using imfilter in the previous step
        
    end
    
end

Img = sum(cat(3,Img_piece_cell{:}),3);    %% sum together the matrices inside Img_piece_cell
end