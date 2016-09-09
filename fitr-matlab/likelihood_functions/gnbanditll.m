%===============================================================================
% GNBANDITLL Constructs a likelihood function from observation
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

classdef gnbanditll
    methods (Static)

        function loglik = lrbeta(params, data)

            T      = size(data.S, 1);
            Q      = zeros(length(unique(data.S)), length(unique(data.A)));
            loglik = 0;
            alpha  = params(1);
            beta   = params(2);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));

                RPE     = r - Q(s, a);
                Q(s, a) = Q(s, a) + alpha*RPE;

            end
        end

        function loglik = lrbetarho(params, data)

            T      = size(data.S, 1);
            Q      = zeros(length(unique(data.S)), length(unique(data.A)));
            loglik = 0;
            alpha  = params(1);
            beta   = params(2);
            rho    = params(3);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));

                RPE     = rho*r - Q(s, a);
                Q(s, a) = (1-alpha)*Q(s, a);
                Q(s, a) = Q(s, a) + alpha*RPE;

            end
        end

        function loglik = lr2beta(params, data)

            T      = size(data.S, 1);
            Q      = zeros(length(unique(data.S)), length(unique(data.A)));
            loglik = 0;
            alphapos  = params(1);
            alphaneg  = params(2);
            beta      = params(3);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));

                RPE     = r - Q(s, a);
                if RPE < 0
                    Q(s, a) = Q(s, a) + alphaneg*RPE;
                else
                    Q(s, a) = Q(s, a) + alphapos*RPE;
                end

            end
        end

        function loglik = lr2betarho(params, data)

            T      = size(data.S, 1);
            Q      = zeros(length(unique(data.S)), length(unique(data.A)));
            loglik = 0;
            alphapos  = params(1);
            alphaneg  = params(2);
            beta      = params(3);
            rho       = params(4);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));

                RPE     = rho*r - Q(s, a);
                if RPE < 0
                    Q(s, a) = Q(s, a) + alphaneg*RPE;
                else
                    Q(s, a) = Q(s, a) + alphapos*RPE;
                end

            end
        end

        function loglik = randmodel(params, data)

            T      = size(data.S, 1);
            Q      = zeros(length(unique(data.S)), length(unique(data.A)));
            loglik = 0;
            beta   = params(1);

            for t = 1:T
                s       = data.S(t);
                a       = data.A(t);
                r       = data.R(t);
                loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));
            end
        end

    end
end
