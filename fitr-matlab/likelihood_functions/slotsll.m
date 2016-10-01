%===============================================================================
% SLOTSLL Constructs a likelihood function from observation
%   model and learning model and returns log likelihood of the data given
%   the model parameters
%
% INPUTS:
%   params     = current parameter estimates
%   data       = structure of single participant data with fields
%                .S = [T by 1] state vector  (T=trials)
%                .A = [T by 1] action vector
%                .R = [T by 1] reward vector
%
% OUTPUT:
%   logLik = log likelihood of the data given the model parameters
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada.
%===============================================================================

classdef slotsll
    methods (Static)

        function loglik = lrbeta(params, data)
            T      = size(data.S, 1);
            Q      = zeros(1, 4);
            loglik = 0;
            alpha  = params(1);
            beta   = params(2);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(1, a) - logsumexp(beta.*Q(1,:));

                RPE     = r - Q(s, a);
                Q(s, a) = Q(s, a) + alpha*RPE;

            end
        end

    end
end
