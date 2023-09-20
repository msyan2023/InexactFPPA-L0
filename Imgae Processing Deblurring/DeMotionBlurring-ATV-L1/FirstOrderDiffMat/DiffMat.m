function [Ux, Uy]=DiffMat(Img)
% This function computes the first order difference matrix applying on the matrix "Img" in
% vertical and horizontal directions.
%
% boundary condition: symmetric(mirror) padding     x1 | x1 x2 ... xn | xn

Ux=diff([Img(1,:); Img],1,1);  %% difference of row
Uy=diff([Img(:,1) Img],1,2);   %% difference of column

end