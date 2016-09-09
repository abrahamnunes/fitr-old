%===============================================================================
%TWOSTEPLL Is a set of likelihood functions for fitting the twostep task
%
% INPUTS:
%   params     = current parameter estimates
%   data       = structure of single participant data with fields
%                .S = [T by 2] state vector  (T=trials)
%                .A = [T by 2] action vector
%                .R = [T by 1] reward vector
%
% OUTPUT:
%   loglik = log likelihood of the data given the model parameters
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada.
%===============================================================================

classdef twostepll
    methods (Static)

    function loglik = lrbetaomega(params, data)

        ntrials = size(data.S, 1);

        alpha = params(1);
        beta  = params(2);
        omega = params(3);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);
        Qb = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            Q = omega.*Qb + (1-omega).*Qf;
        end

    end

    function loglik = lrbetalambdaomega(params, data)

        % Setup variables
        T      = size(data.S, 1);
        Q      = zeros(length(unique(data.S)), length(unique(data.A)));
        Qf     = zeros(length(unique(data.S)), length(unique(data.A)));
        Qmb    = zeros(length(unique(data.S)), length(unique(data.A)));

        loglik = 0;
        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);
        omega  = params(4);

        for t = 1:T
            s1       = data.S(t,1); % First state
            s2       = data.S(t,2); % Second state
            a1       = data.A(t,1); % First action
            a2       = data.A(t,2); % Second action
            r        = data.R(t);   % Reward at trial

            % Compute current log likelihood for the trial
            loglik  = loglik + beta*Q(s1,a1)  - logsumexp(beta.*Q(s1,:));
            loglik  = loglik + beta*Q(s2,a2) - logsumexp(beta.*Q(s2,:));

            % Learn model free
            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            % Learn model based
            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            % Combine model-based and model-free value estimates
            Q = omega*Qmb + (1-omega)*Qf;

        end

    end

    function loglik = lrbetarhoomega(params, data)

        % Setup variables
        T      = size(data.S, 1);
        Q      = zeros(length(unique(data.S)), length(unique(data.A)));
        Qf     = zeros(length(unique(data.S)), length(unique(data.A)));
        Qmb    = zeros(length(unique(data.S)), length(unique(data.A)));

        loglik = 0;
        alpha  = params(1);
        beta   = params(2);
        rho    = params(3);
        omega  = params(4);

        for t = 1:T
            s1       = data.S(t,1); % First state
            s2       = data.S(t,2); % Second state
            a1       = data.A(t,1); % First action
            a2       = data.A(t,2); % Second action
            r        = data.R(t);   % Reward at trial

            % Compute current log likelihood for the trial
            loglik  = loglik + beta*Q(s1,a1)  - logsumexp(beta.*Q(s1,:));
            loglik  = loglik + beta*Q(s2,a2) - logsumexp(beta.*Q(s2,:));

            % Learn model free
            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = rho*r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            % Learn model based
            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            % Combine model-based and model-free value estimates
            Q = omega*Qmb + (1-omega)*Qf;

        end

    end

    function loglik = lrbetalambdarhoomega(params, data)

        % Setup variables
        T      = size(data.S, 1);
        Q      = zeros(length(unique(data.S)), length(unique(data.A)));
        Qf     = zeros(length(unique(data.S)), length(unique(data.A)));
        Qmb    = zeros(length(unique(data.S)), length(unique(data.A)));

        loglik = 0;
        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);
        rho    = params(4);
        omega  = params(5);

        for t = 1:T
            s1       = data.S(t,1); % First state
            s2       = data.S(t,2); % Second state
            a1       = data.A(t,1); % First action
            a2       = data.A(t,2); % Second action
            r        = data.R(t);   % Reward at trial

            % Compute current log likelihood for the trial
            loglik  = loglik + beta*Q(s1,a1)  - logsumexp(beta.*Q(s1,:));
            loglik  = loglik + beta*Q(s2,a2) - logsumexp(beta.*Q(s2,:));

            % Learn model free
            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = rho*r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            % Learn model based
            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            % Combine model-based and model-free value estimates
            Q = omega*Qmb + (1-omega)*Qf;

        end

    end

    function loglik = mb_lrbeta(params, data)

        ntrials = size(data.S, 1);

        alpha = params(1);
        beta  = params(2);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);
        Qb = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            Q = Qb;
        end

    end

    function loglik = mb_lrbetarho(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        rho    = params(3);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);
        Qb = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe        = rho*r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            Q = Qb;
        end

    end

    function loglik = mb_lrbetalambda(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);
        Qb = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            Q = Qb;
        end

    end

    function loglik = mb_lrbetalambdarho(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);
        rho    = params(4);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);
        Qb = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            Qb(1,:) = ([0.7 0.3; 0.3 0.7]*max(Qf(2:3,:), [], 2))';

            Q = Qb;
        end

    end

    function loglik = mf_lrbeta(params, data)

        ntrials = size(data.S, 1);

        alpha = params(1);
        beta  = params(2);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            Q = Qf;
        end

    end

    function loglik = mf_lrbetarho(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        rho    = params(3);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe        = rho*r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe;

            Q = Qf;
        end

    end

    function loglik = mf_lrbetalambda(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            Q = Qf;
        end

    end

    function loglik = mf_lrbetalambdarho(params, data)

        ntrials = size(data.S, 1);

        alpha  = params(1);
        beta   = params(2);
        lambda = params(3);
        rho    = params(4);

        Q  = zeros(3, 2);
        Qf = zeros(3, 2);

        loglik = 0;

        for t = 1:ntrials
            s1 = data.S(t, 1);
            s2 = data.S(t, 2);
            a1 = data.A(t, 1);
            a2 = data.A(t, 2);
            r  = data.R(t);

            loglik = loglik + beta.*Q(s1, a1) - logsumexp(beta.*Q(s1,:));
            loglik = loglik + beta.*Q(s2, a2) - logsumexp(beta.*Q(s2,:));

            Qf(s1, a1) = Qf(s1, a1) + alpha*(Qf(s2, a2) - Qf(s1, a1));
            rpe = r - Qf(s2, a2);
            Qf(s2, a2) = Qf(s2, a2) + alpha*rpe;
            Qf(s1, a1) = Qf(s1, a1) + alpha*rpe*lambda;

            Q = Qf;
        end

    end

    function loglik = randmodel(params, data)

        % Setup variables
        T      = size(data.S, 1);
        Q      = zeros(length(unique(data.S)), length(unique(data.A)));
        Qf     = zeros(length(unique(data.S)), length(unique(data.A)));
        Qmb    = zeros(length(unique(data.S)), length(unique(data.A)));

        loglik = 0;
        beta   = params(1);

        for t = 1:T
            s1       = data.S(t,1); % First state
            s2       = data.S(t,2); % Second state
            a1       = data.A(t,1); % First action
            a2       = data.A(t,2); % Second action
            r        = data.R(t);   % Reward at trial

            % Compute current log likelihood for the trial
            loglik  = loglik + beta*Q(s1,a1)  - logsumexp(beta.*Q(s1,:));
            loglik  = loglik + beta*Qf(s2,a2) - logsumexp(beta.*Qf(s2,:));

        end

    end

    end
end
