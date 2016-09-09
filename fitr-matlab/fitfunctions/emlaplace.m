%===============================================================================
%EMLAPLACE Estimates hyperparameters of a reinforcement learning model
% by using Expectation-Maximiation and the Laplace Approximation
%
% INPUTS:
%   The `model` structure passed through the `OptimizeParams` function
%
% OUTPUTS:
%   The `model` structure with updated hyperparameters
%
% CITATION
%   Huys, Q. J. M., Cools, R., G�lzer, M.,et al. (2011). Disentangling the
%    roles of approach, activation and valence in instrumental and
%    pavlovian responding. PLoS Computational Biology, 7(4).
%
% 2016 Abraham Nunes; Dalhousie University. Halifax, NS
%===============================================================================

function fit = emlaplace(fit)

% Find the mean parameter estimates across all subjects
phiMu  = mean(fit.params, 1);
phiVar = zeros(fit.K, fit.K);

for i = 1:fit.N
    mapEst = fit.params(i, :);
    invHess = inv(fit.H{i});
    phiVar = phiVar + ( (mapEst'*mapEst + invHess) - (phiMu'*phiMu) );
end

fit.hparams.mu    = phiMu;
fit.hparams.sigma = phiVar/fit.N;

end
