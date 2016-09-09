%===============================================================================
%DIRICHLETRAND Samples an n dimensional vector of values from the Dirichlet
% distribution with parameters alpha
%
% INPUTS:
%   alpha = [1 by K] vector of Dirichlet parameters (K = number of classes)
%   n     = number of samples to draw
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS
%===============================================================================

function sample = dirichletrand(alpha, n)

K      = length(alpha); % Number of classes
sample = gamrnd(repmat(alpha, n, 1), 1, [n, K]);
sample = sample ./ repmat(sum(sample,2), 1, K);

end
