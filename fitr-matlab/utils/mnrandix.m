%===============================================================================
%INDEXEDMNSAMPLE Samples from a multinomial distribution and returns the
%   indexed value of the sampled item, rather than the vector
%
% INPUTS:
%   n = number of trials
%   p = [1 by K] array of probabilities (must sum to 1)
%   m = number of samples
%
% OUTPUTS:
%   sample = [M by 1] array of samples
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada
%===============================================================================

function sample = mnrandix(n, p, m)

    [~, sample] = max(mnrnd(n, p, m), [] ,2);

end
