function y = sqHinge(x)
%% square hinge loss function
y = 1/2*(max(1-x,0)).^2;
end