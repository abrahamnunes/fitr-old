%===============================================================================
%PARAMTRANSFORMATION Transforms variables from constrained to
% unconstrained spaces, and vice versa
%
% INPUTS:
%   param = [1 by K] vector of parameter values
%   rng   = {1 by K} array of strings denoting the range of the parameter
%       - 'unit' = [0, 1]
%       - 'pos'  = [0, infty]
%       - 'unc'  = [-infty, +infty]
%   transformType = a string denoting the type of transformation
%       - 'UC' = unconstrained to constrained space
%       - 'CU' = constrained to unconstrained space
%
% OUTPUTS:
%   params = the transformed parameter vector values
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada
%===============================================================================

function param = paramtransform(param, rng, transformType)

K = length(param);
for k = 1:K
    switch transformType
        case 'UC'
            switch rng{k}
                case 'unit'
                    if param(k) < -16.
                        param(k) = -16.;
                    end
                    param(k) = 1./(1+exp(-param(k)));
                case 'pos'
                    if param(k) > 16.0
                        param(k) = 16.0;
                    end
                    param(k) = exp(param(k));
                otherwise
                    param(k) = param(k);
            end
        case 'CU'
            switch rng{k}
                case 'unit'
                    param(k) = -log((1/param(k)) - 1);
                case 'pos'
                    param(k) = log(param(k));
                otherwise
                    param(k) = param(k);
            end
        otherwise
            param(k) = param(k);
    end

end
