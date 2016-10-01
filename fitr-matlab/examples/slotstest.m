% DEMO USING SLOTS TASK (N-ARMED BANDIT WITH GAUSSIAN REWARDS)
%
% (c)2016 Abraham Nunes; Dalhousie University, Halifax, NS, Canada
%===============================================================================
%
%   CREATE A GROUP OF SUBJECTS AND RUN THROUGH THE TASK
%
%===============================================================================

subjects.N           = 50;
subjects.params      = zeros(subjects.N, 2);
subjects.params(:,1) = betarnd(1.1, 1.1, [subjects.N, 1]);
subjects.params(:,2) = gamrnd(5, 1, [subjects.N, 1]);

taskparams.ntrials  = 100;
taskparams.nactions = 4;    %number of armed bandit

results = slots.vanilla(subjects, taskparams);

%===============================================================================
%
%   CREATE MODEL
%
%===============================================================================

model.lik      = @slotsll.lrbeta;
model.param    = rlparam.learningrate();
model.param(2) = rlparam.inversetemp();

%===============================================================================
%
%   FIT MODEL
%
%===============================================================================

fitoptions.maxiters   = 1000;
fitoptions.nstarts    = 2;
fitoptions.climit     = 10;

fit = fitmodel(results, model, fitoptions);

%===============================================================================
%
%   PLOT RESULTS (uses the `pqplot` package)
%
%===============================================================================

figure(); %Plot an example of the reward paths
plot(1:size(results(1).rpaths, 1), results(1).rpaths, ...
     'LineWidth', 1.5);
title('Reward Paths');
xlabel('Trial');
ylabel('Reward');

figure();
subplot(1, 3, 1);
p = pqscatter(subjects.params(:,1), fit.params(:,1), {'match'});
pqtitle(p, 'Learning Rate');
xlabel('Actual'); ylabel('Estimate');

subplot(1, 3, 2);
p = pqscatter(subjects.params(:,2), fit.params(:,2), {'match'});
pqtitle(p, 'Inverse Temperature');
xlabel('Actual'); ylabel('Estimate');

subplot(1, 3, 3);
x = paramtransformvect(fit.params(:,1), 'unit', 'CU');
y = paramtransformvect(fit.params(:,2), 'pos', 'CU');
p = pqscatter(x, y, {'lm'});
pqtitle(p, 'Estimate Correlation');
xlabel('\alpha'); ylabel('\beta');
