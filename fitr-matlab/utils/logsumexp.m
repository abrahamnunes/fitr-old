%===============================================================================
%LOGSUMEXP A general log-sum-exp function that accepts only one argument
% which is then less restrictive on the form of the policy functions
%
%   INPUT:
%       Q    = State-action value table (selected at a current state)
%       beta = Inverse temperature parameter
%
%   OUTPUT:
%       y    = log-sum-exp value
%
% CREDITS: Based on the function found in Samuel Gershman's `mfit` package
%
% 2016 Abraham Nunes
%===============================================================================

function y = logsumexp( x )

maxQa = max(x);
Qcor  = x - maxQa;
y     = maxQa + log(sum(exp(Qcor)));
i     = find(~isfinite(maxQa));
if ~isempty(i)
    y(i) = maxQa(i);
end


end
