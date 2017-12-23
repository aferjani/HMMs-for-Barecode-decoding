function [lpp] = state_postdict(A, log_p)
% STATE_POSTDICT Computes A’*p in log domain
%
% [lpp] = state_postdict(A, log_p)
%
% Inputs :
% A : State transition matrix
% log_p : log p(y_{k+1:K}|x_{k+1}) Updated potential
%
% Outputs :
% lpp : log p(y_{k+1:K}| x_k) Postdicted potential
mx = max(log_p(:)); % Stable computation
p = exp(log_p - mx);
lpp = log(A'*p + eps) + mx; % add eps for numerical instability