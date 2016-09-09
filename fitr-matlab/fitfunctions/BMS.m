%==============================================================================
%BMS Implements the Bayesian model selection procedure
% outlined by:
%   Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014).
%    Bayesian model selection for group studies - Revisited.
%    NeuroImage, 84, 971�985.
%
% This script is based much upon Samuel Gershman's MFIT toolbox available
% on GitHub
%
% INPUTS:
%   models = structure containing the various fits
%
% OUTPUTS:
%   bms    = structure containing the results of Bayesian model selection
%       .pms = probability of model M given subject
%       .r   = expected probabilty of each model in population
%       .xp  = exceedance probability
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS
%==============================================================================

function bms = BMS(models)

nModels   = size(models, 2);
nSubjects = size(models(1).fit.params, 1);
predError = 1; convergenceLimit = 10e-10;

% Initialize output structure
bms = struct();
for m = 1:nModels
    bms.modelNames{m} = models(m).name;
end

% Preallocate memory
u    = zeros(nSubjects, nModels);
g    = zeros(nSubjects, nModels);
a    = zeros(nSubjects, nModels);
LME  = zeros(nSubjects, nModels);


% Initialize Dirichlet priors
alpha0 = ones(1, nModels);
alpha  = alpha0;

iterCount = 1;

disp('----BAYESIAN MODEL SELECTION----');
while predError > convergenceLimit
    disp(['BMS Iteration ', num2str(iterCount)]);

    for i = 1:nSubjects
        log_u = zeros(1, nModels);
        for m = 1:nModels
            testLME = models(m).fit.sLME(i);

            if isnan(testLME) || isinf(testLME) || ~isreal(testLME); % use BIC if Hessian is degenerate
                LME(i, m) = -0.5*(models(m).fit.sBIC(i) - models(m).fit.K*log(2*pi));
            else
                LME(i, m) = models(m).fit.sLME(i);
            end

            log_u(m) = LME(i, m) + psi(alpha(m)) - psi(sum(alpha));
        end
        u(i,:) = exp(log_u - max(log_u));
        g(i,:) = u(i,:)./sum(u(i,:));
        a(i,:) = mnrnd(1, g(i,:), 1);
    end

    beta          = sum(a);
    alphaPrevious = alpha;
    alpha         = alpha0 + beta;

    %Test for convergence
    predError = norm(alpha - alphaPrevious);

    %Update counter
    iterCount = iterCount + 1;
end

% Find expected values of model probabilities
bms.pms     = g;
bms.r       = alpha ./ sum(alpha);
bms.xp      = dirichlet_exceedance(alpha);

posterior.a = alpha;
posterior.r = g';
priors.a    = alpha0;
bor = BMS_bor(LME',posterior, priors);
bms.pxp = bms.xp*(1-bor) + bor/nModels;

disp('----MODEL SELECTION COMPLETE----');
end

function xp = dirichlet_exceedance(alpha)
    % This script is taken from Samuel Gershman's MFIT toolbox available
    % on GitHub
    % Compute exceedance probabilities for a Dirichlet distribution

    Nsamp = 1e6;

    Nk = length(alpha);

    % Perform sampling in blocks
    %--------------------------------------------------------------------------
    blk = ceil(Nsamp*Nk*8 / 2^28);
    blk = floor(Nsamp/blk * ones(1,blk));
    blk(end) = Nsamp - sum(blk(1:end-1));

    xp = zeros(1,Nk);
    for i=1:length(blk)

        % Sample from univariate gamma densities then normalise
        % (see Dirichlet entry in Wikipedia or Ferguson (1973) Ann. Stat. 1,
        % 209-230)
        %----------------------------------------------------------------------
        r = zeros(blk(i),Nk);
        for k = 1:Nk
            r(:,k) = gamrnd(alpha(k),1,blk(i),1);
        end
        sr = sum(r,2);
        for k = 1:Nk
            r(:,k) = r(:,k)./sr;
        end

        % Exceedance probabilities:
        % For any given model k1, compute the probability that it is more
        % likely than any other model k2~=k1
        %----------------------------------------------------------------------
        [~, j] = max(r,[],2);
        xp = xp + histc(j, 1:Nk)';

    end
    xp = xp / Nsamp;
end

function [bor,F0,F1] = BMS_bor(L,posterior,priors,C)
    % This script is taken from Samuel Gershman's MFIT toolbox available
    % on GitHub
    % Compute Bayes Omnibus Risk

    if nargin < 4
        options.families = 0;
        % Evidence of null (equal model freqs)
        F0 = FE_null(L,options);
    else
        options.families = 1;
        options.C = C;
        % Evidence of null (equal model freqs) under family prior
        [~,F0] = FE_null(L,options);
    end

    % Evidence of alternative
    F1 = FE(L,posterior,priors);

    % Implied by Eq 5 (see also p39) in Rigoux et al.
    % See also, last equation in Appendix 2
    bor = 1/(1+exp(F1-F0));
end

function [F,ELJ,Sqf,Sqm] = FE(L,posterior,priors)
    % This script is taken from Samuel Gershman's MFIT toolbox available
    % on GitHub
    % derives the free energy for the current approximate posterior
    % This routine has been copied from the VBA_groupBMC function
    % of the VBA toolbox http://code.google.com/p/mbb-vb-toolbox/
    % and was written by Lionel Rigoux and J. Daunizeau
    %
    % See equation A.20 in Rigoux et al. (should be F1 on LHS)

    [K,n] = size(L);
    a0 = sum(posterior.a);
    Elogr = psi(posterior.a) - psi(sum(posterior.a));
    Sqf = sum(gammaln(posterior.a)) - gammaln(a0) - sum((posterior.a-1).*Elogr);
    Sqm = 0;
    for i=1:n
        Sqm = Sqm - sum(posterior.r(:,i).*log(posterior.r(:,i)+eps));
    end
    ELJ = gammaln(sum(priors.a)) - sum(gammaln(priors.a)) + sum((priors.a-1).*Elogr);
    for i=1:n
        for k=1:K
            ELJ = ELJ + posterior.r(k,i).*(Elogr(k)+L(k,i));
        end
    end
    F = ELJ + Sqf + Sqm;
end


function [F0m,F0f] = FE_null (L,options)
    % This script is taken from Samuel Gershman's MFIT toolbox available
    % on GitHub
    % Free energy of the 'null' (H0: equal frequencies)
    %
    % F0m       Evidence for null (ie. equal probs) over models
    % F0f       Evidence for null (ie. equal probs) over families
    %
    % This routine derives from the VBA_groupBMC function
    % of the VBA toolbox http://code.google.com/p/mbb-vb-toolbox/
    % written by Lionel Rigoux and J. Daunizeau
    %
    % See Equation A.17 in Rigoux et al.

    [K,n] = size(L);
    if options.families
        f0 = options.C*sum(options.C,1)'.^-1/size(options.C,2);
        F0f = 0;
    else
        F0f = [];
    end
    F0m = 0;
    for i=1:n
        tmp = L(:,i) - max(L(:,i));
        g = exp(tmp)./sum(exp(tmp));
        for k=1:K
            F0m = F0m + g(k).*(L(k,i)-log(K)-log(g(k)+eps));
            if options.families
                F0f = F0f + g(k).*(L(k,i)-log(g(k)+eps)+log(f0(k)));
            end
        end
    end
end
