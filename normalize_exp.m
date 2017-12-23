function [p] = normalize_exp(log_p)
%This function returns the normalized exp of a given log vector
a = max(log_p(:));
A = exp(log_p - a);
Z = sum(A); 
Z(Z == 0) = 1;
p = A./Z;
end
