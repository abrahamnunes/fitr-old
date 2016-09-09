% Testing ground for including gradients with likelihood functions

function [loglik, g] = gradientlik(params, data)
    T      = size(data.S, 1);
    Q      = zeros(length(unique(data.S)), length(unique(data.A)));
    loglik = 0;
    alpha  = params(1);
    beta   = params(2);

    %Initialize Q gradients
    dQa = zeros(size(Q));
    dLa = 0;
    dLb = 0;

    for t = 1:T

        s       = data.S(t);
        a       = data.A(t);
        r       = data.R(t);

        % Update dQ/dlr
        dQa(s, a) = (1-alpha)*dQa(s, a) + (r - Q(s, a));

        % Compute log likelihood
        loglik  = loglik + beta.*Q(s, a) - logsumexp(beta.*Q(s,:));

        RPE     = r - Q(s, a);
        Q(s, a) = Q(s, a) + alpha*RPE;

        % Update log likelihood gradients
        paprime = softmx(beta*Q(s,:));
        dLa = dLa + (beta*dQa(s, a) - sum(paprime.*(beta*dQa(s,:))));
        dLb = dLb + (Q(s, a) - sum(paprime.*Q(s,:)));

    end

    loglik = -loglik;
    if nargout > 1
        g = [-dLa; -dLb]; % return gradient
    end
end
