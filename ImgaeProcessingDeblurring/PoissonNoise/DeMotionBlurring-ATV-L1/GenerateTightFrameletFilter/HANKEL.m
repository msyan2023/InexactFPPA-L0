function H = HANKEL(a,b)

%% generate hankel matrix (see the reference paper page 41)

    JJ = length(a);
   
    H = zeros(JJ,JJ);

    H(1,1:end-1) = a(2:end);
    
    H(end,2:end) = flip(b(2:end));
    
    for i=2:JJ-1
        
        H(i,:) = [a(i+1:end) 0 flip(b(JJ-i+2:end))];
        
    end
    
    
end