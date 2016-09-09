%EMLAPLACELOGPOST Computes the unnormalized log-posterior of the
% model parameters when the EMLaplace hyperparameter estimation procedure
% is used
%
% OUTPUTS:
%   logPost = log-posterior probability
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS, Canada.
%===============================================================================

function logpost = emlaplacelogpost(params,hparams,rng,model,data)

paramsT = paramtransform(params, rng, 'UC');
logpost = model.lik(paramsT, data) + log(hparams.priorpdf(params, hparams.mu, hparams.sigma));

end
