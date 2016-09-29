%===============================================================================
%
%   GENERATE DATA
%
%===============================================================================

% Task parameters
taskparams.ntrials  = 200;
taskparams.nstates  = 2;
taskparams.nactions = 2;
taskparams.preward  = [0.7, 0.3; 0.3, 0.7];
taskparams.rewards  = [1, -1; -1, 1];
taskparams.ptrans   = [0.5; 0.5];

% First generate some subjects
subjects.N           = 50;
subjects.params      = zeros(subjects.N, 2);
subjects.params(:,1) = betarnd(1.1, 1.1, [subjects.N, 1]); %learning rate
subjects.params(:,2) = gamrnd(5., 1., [subjects.N, 1]); %inverse temperature

% Now generate some data

results = gonogobandit.vanilla(subjects, taskparams);

% Plot reward Prediction Errors
rpes = reshape([results.rpe], [subjects.N, taskparams.ntrials])';
figure();
g = pqts(rpes, [0, 100], 0.1, {'seriesmean'});
xlabel('Trial');
ylabel('Reward Prediction Error');
pqtitle(g, 'Reward Prediction Error');

%===============================================================================
%
%   CREATE MODELS
%
%===============================================================================

model1.lik      = @gnbanditll.lrbeta;
model1.param    = rlparam.learningrate();
model1.param(2) = rlparam.inversetemp();

model2.lik      = @gnbanditll.lrbetarho;
model2.param    = rlparam.learningrate();
model2.param(2) = rlparam.inversetemp();
model2.param(3) = rlparam.rewardsensitivity();

model3.lik      = @gnbanditll.lr2beta;
model3.param    = rlparam.learningrate();
model3.param(2) = rlparam.learningrate();
model3.param(3) = rlparam.inversetemp();

model4.lik      = @gnbanditll.lr2betarho;
model4.param    = rlparam.learningrate();
model4.param(2) = rlparam.learningrate();
model4.param(3) = rlparam.inversetemp();
model4.param(4) = rlparam.rewardsensitivity();

model5.lik      = @gnbanditll.randmodel;
model5.param    = rlparam.inversetemp();

%===============================================================================
%
%   FIT PARAMETERS
%
%===============================================================================

fitoptions.maxiters   = 1000;
fitoptions.nstarts    = 2;
fitoptions.climit     = 10;

fit1 = fitmodel(results, model1, fitoptions);
fit2 = fitmodel(results, model2, fitoptions);
fit3 = fitmodel(results, model3, fitoptions);
fit4 = fitmodel(results, model4, fitoptions);
fit5 = fitmodel(results, model5, fitoptions);

%===============================================================================
%
%   BAYESIAN MODEL SELECTION
%
%===============================================================================

models(1).name = 'Model 1';
models(1).fit  = fit1;

models(2).name = 'Model 2';
models(2).fit  = fit2;

models(3).name = 'Model 3';
models(3).fit  = fit3;

models(4).name = 'Model 4';
models(4).fit  = fit4;

models(5).name = 'Model 5';
models(5).fit  = fit5;

bms = BMS(models);

%===============================================================================
%
%   PLOTS OF FITTED MODELS
%
%===============================================================================

% Copy parameter vectors to have simpler names (for readability)
salpha  = subjects.params(:,1); %subject alpha
sbeta   = subjects.params(:,2); %subject beta
falpha  = horzcat(fit1.params(:,1), fit2.params(:,1), fit3.params(:,1), fit4.params(:,1)); %fitted model alphas
fbeta   = horzcat(fit1.params(:,2), fit2.params(:,2), fit3.params(:,3), fit4.params(:,3)); %fitted model betas

% Parameter scatterplots
figure();
subplot(2, 1, 1);
g1 = pqscatter(subjects.params(:,1), horzcat(fit1.params(:,1),    fit2.params(:,1), fit3.params(:,1), fit4.params(:,1)), {'match'});
xlabel('Actual');
ylabel('Estimate');
pqtitle(g1, 'Learning Rate (\alpha)');

subplot(2, 1, 2);
g2 = pqscatter(subjects.params(:,2), horzcat(fit1.params(:,2), fit2.params(:,2), fit3.params(:,3), fit4.params(:,3)), {'match'});
xlabel('Actual');
ylabel('Estimate');
pqtitle(g2, 'Inverse Temperature (\beta)');

% Parameter line plots
ptu = @(x) paramtransform(x, {'unit'}, 'CU');
ptp = @(x) paramtransform(x, {'pos'}, 'CU');

figure();
subplot(4, 2, 1);
g1 = pqline(1:subjects.N, arrayfun(ptu, subjects.params(:,1))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptu, fit1.params(:,1)), fit1.errs(:,1)); hold off;
pqtitle(g2, '\alpha');
ylabel('Model 1');
xlim([0, subjects.N+1]);

subplot(4, 2, 2);
g1 = pqline(1:subjects.N, arrayfun(ptp, subjects.params(:,2))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptp, fit1.params(:,2)), fit1.errs(:,2)); hold off;
pqtitle(g2, '\beta');
xlim([0, subjects.N+1]);

subplot(4, 2, 3);
g1 = pqline(1:subjects.N, arrayfun(ptu, subjects.params(:,1))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptu, fit2.params(:,1)), fit1.errs(:,1)); hold off;
ylabel('Model 2');
xlim([0, subjects.N+1]);

subplot(4, 2, 4);
g1 = pqline(1:subjects.N, arrayfun(ptp, subjects.params(:,2))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptp, fit2.params(:,2)), fit1.errs(:,2)); hold off;
xlim([0, subjects.N+1]);

subplot(4, 2, 5);
g1 = pqline(1:subjects.N, arrayfun(ptu, subjects.params(:,1))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptu, fit3.params(:,1)), fit1.errs(:,1)); hold off;
ylabel('Model 3');
xlim([0, subjects.N+1]);

subplot(4, 2, 6);
g1 = pqline(1:subjects.N, arrayfun(ptp, subjects.params(:,2))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptp, fit3.params(:,2)), fit1.errs(:,2)); hold off;
xlim([0, subjects.N+1]);

subplot(4, 2, 7);
g1 = pqline(1:subjects.N, arrayfun(ptu, subjects.params(:,1))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptu, fit4.params(:,1)), fit1.errs(:,1)); hold off;
ylabel('Model 4');
xlabel('Subject');
xlim([0, subjects.N+1]);

subplot(4, 2, 8);
g1 = pqline(1:subjects.N, arrayfun(ptp, subjects.params(:,2))); hold on;
g2 = pqerrorbar(1:subjects.N, arrayfun(ptp, fit4.params(:,2)), fit1.errs(:,2)); hold off;
xlabel('Subject');
xlim([0, subjects.N+1]);

suptitle('Actual-Estimate Plots for Model Parameters');

% Parameter correlograms
%figure();
%subplot(2, 1, 1);
%g1 = pqcorrelogram(horzcat(salpha, falpha), {'Subjects', 'Model 1', 'Model 2', 'Model 3', 'Model 4'});
%pqtitle(g1, 'Correlations (\alpha)');

%subplot(2, 1, 2);
%g2 = pqcorrelogram(horzcat(sbeta, fbeta), {'Subjects', 'Model 1', 'Model 2', 'Model 3', 'Model 4'});
%pqtitle(g2, 'Correlations (\beta)');

% Plot model fitting and selection results
figure();
subplot(3, 2, 1);
g1 = pqbar({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'}, [sum(fit1.sAIC) sum(fit2.sAIC) sum(fit3.sAIC) sum(fit4.sAIC) sum(fit5.sAIC)]);
pqtitle(g1, 'Aikake Information Criterion');

subplot(3, 2, 2);
g2 = pqbar({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'}, [sum(fit1.sBIC) sum(fit2.sBIC) sum(fit3.sBIC) sum(fit4.sBIC) sum(fit5.sBIC)]);
pqtitle(g2, 'Bayesian Information Criterion');

subplot(3, 2, 3);
g3 = pqbar({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'}, bms.xp);
pqtitle(g3, 'Exceedance Probabilities');

subplot(3, 2, 4);
g4 = pqbar({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'}, bms.pxp);
pqtitle(g4, 'Protected Exceedance Probabilities');

subplot(3, 2, 5);
g5 = pqbar({'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model5'}, [fit1.pp, fit2.pp, fit3.pp, fit4.pp, fit5.pp]);
pqtitle(g5, 'Total Predictive Probability');
