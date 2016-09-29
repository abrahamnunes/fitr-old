%==============================================================================
% OPTIMIZEPARAMS Finds the MAP estimate of parameters for reinforcement
% learning model given participants' data
%
% INPUTS: TO BE COMPLETED...
%
%
% OUTPUTS:
%   fit = structure containing the fit data
%           .fit.N         = Number of subjects in sample
%           .fit.T         = Number of trials per subject
%           .fit.K         = Number of parameters in model
%           .fit.params    = [N by K] array of MAP parameter estimates
%           .fit.loglik    = [N by 1] log likelihood of data | par
%           .fit.logpost   = [N by 1] log post prob of data | par, hyperpar
%           .fit.H         = {1 by N} cell array of Hessians from parameter
%                             estimation
%           .fit.errs      = standard deviation of parameter estimates
%                            computed from Hessian
%          $.fit.hparamss  = [1 by K] structure of hyperparameter estimates
%                          .fit.hparamss(k).est = vector of hparam estimates
%           .fit.sBIC      = [N by 1] Subject level Bayesian Information
%                             Criterion
%           .fit.sAIC      = [N by 1] Subject level Aikake Information
%                             Criterion
%           .fit.sLME = [N by 1] model evidence at the subject level
%                             used to compute overall model evidence
%           .fit.LME  = overall model evidence (real). Computed by
%                             summing subject-level model evidence
%           .fit.pp   = total predictive probability (Huys et al. 2011)
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS.
%==============================================================================

function fit = fitmodel(results, model, fitoptions)

    nsubjects = size(results, 2);
    rng       = {model.param.rng}; %extract parameter ranges for transformation

    % Setup output structure
    fit            = struct();
    fit.N          = nsubjects;
    fit.T          = size(results(1).S, 1);
    fit.K          = size(model.param, 2);
    fit.params     = zeros(nsubjects, fit.K);
    fit.errs       = zeros(nsubjects, fit.K);
    fit.loglik     = zeros(nsubjects, 1);
    fit.logpost    = zeros(nsubjects, 1);

    %fit.hparams          = struct();
    %fit.hparams.logLik   = 0;
    fit.hparams.mu       = zeros(1, fit.K);       % Prior mean
    fit.hparams.sigma    = 3*eye(fit.K);       % Prior cov mtx
    fit.hparams.priorpdf = @(x, Mu, Sigma) mvnpdf(x, Mu, Sigma);

    fit.sBIC       = zeros(nsubjects, 1);
    fit.sAIC       = zeros(nsubjects, 1);
    fit.sLME       = zeros(nsubjects, 1);
    fit.LME        = 0;
    fit.sumlogpost = -1e1000;

    % Set optimizer options
    options = optimset('Display', 'off');
    warning off all;


    %LOOP OVER NUMBER OF HYPERPARAMETER LEVEL ITERATIONS
    oldfit     = fit;
    for iter = 1:fitoptions.maxiters

        %LOOP OVER SUBJECTS
        %   Find MAP estimates of parameters

        for s = 1:nsubjects
            disp(['Iteration ', num2str(iter), ': Fitting Subject ', num2str(s)]);

            %Create a function for the log-posterior distribution
            %   x = parameter estimate
            logpostfx = @(x) -emlaplacelogpost(x, fit.hparams, rng, model, results(s));

            %LOOP OVER NUMBER OF RANDOM STARTING POINTS FOR OPTIMIZER
            for i = 1:fitoptions.nstarts

                pinf = 1;
                while pinf == 1
                    params0 = normrnd(0, 2, [1, fit.K]);
                    pinf = isinf(logpostfx(params0));
                end

                %Optimize
                [params,nLogP,~,~,~,H] = fminunc(logpostfx,params0,options);
                LogP = -nLogP;

                %If log probability of parameters has improved, store results
                if i == 1 || fit.logpost(s) < LogP
                    fit.logpost(s)   = LogP;
                    fit.loglik(s)    = model.lik(paramtransform(params, rng, 'UC'), results(s));
                    fit.params(s,:)  = params;
                    fit.H{s}         = H;
                    fit.errs(s,:)    = sqrt(diag(inv(H))');
                    fit.sLME(s)      = LME(LogP, fit.K, H);
                end

            end

            %Compute subject level BIC and AIC, and store
            fit.sBIC(s) = BIC(fit.K, fit.T, fit.loglik(s));
            fit.sAIC(s) = AIC(fit.K, fit.loglik(s));

        end

        dlogpost = abs(fit.sumlogpost(iter)) - abs(sum(fit.logpost));
        if iter > 1 && dlogpost < fitoptions.climit && dlogpost > 0
            %fit = oldfit;
            for s = 1:nsubjects
                fit.params(s,:) = paramtransform(fit.params(s,:), rng, 'UC');
            end

            % Compute total predictive probability
            fit.pp = exp(sum(fit.loglik)/(fit.T*fit.N));

            return;
        else
            disp(['Log-Posterior Probability = ', num2str(sum(fit.logpost)), '; Iteration ', num2str(iter)]);
            fit.sumlogpost(iter+1) = sum(fit.logpost);
            oldfit     = fit;
            if iter < fitoptions.maxiters
                %Optimize hyperparameters
                disp(['Iteration ', num2str(iter), ': Fitting Hyperparameters']);
                fit = emlaplace(fit);

            end
            fit.LME = sum(fit.sLME);
        end
    end

    %Transform parameter estimates to constrained space
    %for s = 1:nsubjects
    %    fit.params(s,:) = paramtransform(fit.params(s,:), rng, 'UC');
    %end



%==============================================================================
%
% FIT METRICS
%   - BIC, AIC, LME
%
%==============================================================================

function bic = BIC(K, T, loglik)
    bic = K*log(T) - 2*loglik;

function aic = AIC(K, loglik)
    aic = K*2 - 2*loglik;

function lme = LME(logpost, K, H)
    lme = logpost + K/2 * log(2*pi) - log(det(H))/2;
