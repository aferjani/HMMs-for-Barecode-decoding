function s = log_sum_exp(a, dim)
% Returns log(sum(exp(a),dim)) by avoiding numerical instability.
if nargin < 2
  dim = 1;
end

[y, i] = max(a,[],dim); % get max and index of max value in the vector 
dims = ones(1,ndims(a)); % construct a vector of ones (will be used in substraction)
dims(dim) = size(a,dim); % intermediary vector to be used in repmat function
a = a - repmat(y, dims); % create a vector filled with max_value and substract it from each value of vector
s = y + log(sum(exp(a),dim)); % do the sum_log_exp operation
i = find(~isfinite(y)); % deal with the case where the max can be infinity
if ~isempty(i)
  s(i) = y(i);
end