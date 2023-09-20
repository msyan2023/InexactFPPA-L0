function [V] = DiffMat_Transp(Ux,Uy)
% This function serves to implement the transpose of the first order
% difference operator
%
% boundary condition: symmetric(mirror) padding     x1 | x1 x2 ... xn | xn

V=[-Ux(2,:); ...
        Ux(2:(end-1),:)-Ux(3:end,:); ...
        Ux(end,:)]...
        + ... 
  [-Uy(:,2) ...
        Uy(:,2:(end-1))-Uy(:,3:end) ...
        Uy(:,end)];
    
end
 