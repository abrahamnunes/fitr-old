rng(123456);

incgroups = [1 3 5];
incmodels = [1 3 5 7];
nfittrials = twostepfit.nfittrials;

% Get group and model names into cell arrays for plotting

groupnames = {};
for i = 1:length(incgroups)
    groupnames{i} = twostepfit.trial(1).experiment.groups(incgroups(i)).subjects.name;
end

modelnames = {};
for j = 1:length(incmodels)
    modelnames{j} = twostepfit.trial(1).experiment.model(incmodels(j)).name;
end

pxps = zeros(length(incgroups),length(incmodels),nfittrials);
for k = 1:nfittrials
    for i = 1:length(incgroups)
        bms = BMS(twostepfit.trial(k).experiment.groups(incgroups(i)).rlfits(incmodels));
        pxps(i,:,k) = bms.pxp;
    end
end

% Plot protected exceedance probabilities
figure();
for i = 1:length(incgroups)
    subplot(length(incgroups), 1, i);
    s = pqcorrelogram(squeeze(pxps(i,:,:)), {1:10}, 'identity', modelnames);
    ylabel('Model Hypothesis');
    pqtitle(s, groupnames{i});

    if i == 3
        xlabel('Fit Iteration');
    end
end
suptitle('Protected Exceedance Probabilities over Testing Iterations');

figure();
for i = 1:length(incgroups)
    subplot(length(incgroups), 1, i);
    s = pqcorrelogram((squeeze(pxps(i,:,:)) == repmat(max(squeeze(pxps(i,:,:))),4,1)), {1:10}, 'identity', modelnames);
    ylabel('Model Hypothesis');
    pqtitle(s, groupnames{i});

    if i == 3
        xlabel('Fit Iteration');
    end
end
suptitle('Best Fitting Model');

%Plot correlations of paramters from best fitting models

sparamnames = {'\alpha', '\beta', '\omega'};
sparamix    = [1 2 5]; % Alpha beta omega locations for subjects
paramix     = [1 1 1 1; 2 2 2 2; 3 4 0 0];
paramcorrs  = zeros(nfittrials,3,length(incgroups));
p_paramcorrs  = zeros(nfittrials,3,length(incgroups));
mspe   = zeros(nfittrials,3,length(incgroups));
mspe_n = zeros(nfittrials,3,length(incgroups));
mpe    = zeros(nfittrials,3,length(incgroups));

for i = 1:length(incgroups)
    for k = 1:nfittrials
        figure();
        %find best fitting model
        [~, bestmodeli] = max(pxps(i,:,k));
        for param = 1:3
            if param == 2
                pt = @(x) paramtransform(x, {'pos'}, 'CU');
            else
                pt = @(x) paramtransform(x, {'unit'}, 'CU');
            end

            x = twostepfit.trial(k).experiment.groups(incgroups(i)).subjects.params(:,sparamix(param));
            if bestmodeli == 3 && param == 3
                y = ones(size(x));
            else
                y = twostepfit.trial(k).experiment.groups(incgroups(i)).rlfits(bestmodeli).fit.params(:, paramix(param,bestmodeli));
                yerr = twostepfit.trial(k).experiment.groups(incgroups(i)).rlfits(bestmodeli).fit.errs(:, paramix(param,bestmodeli));
            end

            if bestmodeli == 3 || i == 3
                if param < 3
                    subplot(2, 1, param);
                    g  = pqline(1:length(x), arrayfun(pt, x)); hold on;
                    g2 = pqerrorbar(1:length(x), arrayfun(pt, y), yerr); hold off;
                    pqtitle(g, [sparamnames{param}, ' Comparison']);
                    xlabel('Subject');
                    ylabel('Parameter Value');
                end
            else
                subplot(3, 1, param);
                g  = pqline(1:length(x), arrayfun(pt, x)); hold on;
                g2 = pqerrorbar(1:length(x), arrayfun(pt, y), yerr); hold off;
                pqtitle(g, [sparamnames{param}, ' Comparison']);
                xlabel('Subject');
                ylabel('Parameter Value');
            end

            %AE parameter correlations
            [paramcorrs(k, param, i), p_paramcorrs(k, param, i)] = corr(x, y);

            %Mean squared prediction errors
            mspe(k,param,i) = mean((x - y).^2);
            x1 = (x - mean(x))/std(x);
            y1 = (y - mean(y))/std(y);
            mspe_n(k, param, i) = mean((x1 - y1).^2);

            mpe(k, param, i) = mean((x-y)./x);

        end
        suptitle(['True model: ', groupnames{i}, '. Fitted model: ', modelnames{bestmodeli}, '. PXP =', num2str(pxps(i, bestmodeli, k))]);
    end
end

%Compute average correlation by Fisher's method:
Z = atanh(paramcorrs(:));
