function a = softmaxaction(x)
%SOFTMAXACTION Implements a softmax action selection procedure
%
% INPUTS:
%   Q           = [1 by nAction] state-action values
%   inversetemp = inverse temperature parameter(s)
%
% OUTPUTS:
%   a = selected action (integer value 1 <= a <= nAction)
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada

y = exp(x)./sum(exp(x));
a = mnrandix(1, y, 1);


end
