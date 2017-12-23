function [lpp] = state_predict(A, log_p)
% STATE_PREDICT Computes A*p in log domain
%
% [lpp] = state_predict(A, log_p)
%
% Inputs :
% A : State transition matrix
% log_p : log p(x_{k-1}, y_{1:k-1}) Filtered potential
%
% Outputs :
% lpp : log p(x_{k}, y_{1:k-1}); Predicted potential
mx = max(log_p(:)); % Stable computation
p = exp(log_p - mx);
lpp = log(A*p + eps) + mx;