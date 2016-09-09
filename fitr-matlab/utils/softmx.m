function y = softmx(x)
%SOFTMX Implements a softmax probability computation
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada

y = exp(x)./sum(exp(x));


end
