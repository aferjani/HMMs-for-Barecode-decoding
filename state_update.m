function [lup] = state_update(obs, log_p)
% STATE_UPDATE State update in log domain
%
% [lup] = state_update(obs, log_p)
%
% Inputs :
% obs : p(y_k| x_k)
% log_p : log p(x_k, y_{1, k-1})
%
% Outputs :
% lup : log p(x_k, y_{1, k-1}) p(y_k| x_k)
lup = obs(:) + log_p;