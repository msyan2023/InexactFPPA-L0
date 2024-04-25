function T = TOEPLITZ(a,b)
    
%% generate toeplitz matrix (see the reference paper page 41)

    JJ = length(a);
   
    T = zeros(JJ,JJ);
    
    T(1,:) = a;
    
    for  i=2:JJ
        
        T(i,:) = [flip(b(2:i)) a(1:JJ-(i-1))];
        
    end
    
end