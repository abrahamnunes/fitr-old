% This script analyzes the results of the twostepexp.m script

nfittrials = twostepfit.nfittrials;
ngroups    = size(twostepfit.trial(1).experiment.groups, 2);
nmodels    = size(twostepfit.trial(1).experiment.groups(1).rlfits, 2);

best_bic = zeros(ngroups, nmodels, nfittrials);
best_pxp = zeros(ngroups, nmodels, nfittrials);
pxp      = zeros(ngroups, nmodels, nfittrials);
bic      = zeros(ngroups, nmodels, nfittrials);

for k = 1:nfittrials
    data = twostepfit.trial(k).experiment;
    for i = 1:ngroups
        try
            pxp(i, :, k) = data.groups(i).bms.pxp;
            [~,pxpi] = max(data.groups(i).bms.pxp);
            best_pxp(i, pxpi, k) = best_pxp(i, pxpi, k) + 1;
        catch
            best_pxp(i, :, k) = NaN;
        end

        modelbic = zeros(1, nmodels);
        try
            for j = 1:nmodels
                modelbic(1,j) = sum(data.groups(i).rlfits(j).fit.sBIC);
            end
            bic(i,:,k) = modelbic;
            [~,bici] = min(modelbic);
            best_bic(i, bici, k) = best_bic(i, bici, k) + 1;
        catch
            best_bic(i, :, k) = NaN;
        end

    end
end

% Get group and model names into cell arrays for plotting

groupnames = {};
for i = 1:ngroups
    groupnames{i} = twostepfit.trial(1).experiment.groups(i).subjects.name;
end

modelnames = {};
for j = 1:nmodels
    modelnames{j} = twostepfit.trial(1).experiment.model(j).name;
end

%===============================================================================
%
%   ACCURACY IN SELECTING PLAUSIBLE VS RANDOM MODEL
%
%===============================================================================

figure();
for i = 1:ngroups
    subplot(3, 2, i);
    s = pqbar(1:10, squeeze(pxp(i,:,:))');
    ylim([0,1]);
    pqtitle(s, groupnames{i});

    if ~isempty(find([1, 3, 5], i))
        ylabel('PEx Probability');
    end

    if i >=5
        xlabel('Fit Iteration');
    end

end
suptitle('Protected Exceedance Probabilities over Trials');

% Find and plot correlations between parameter values for best fit models

%sparamnames = {'\alpha', '\beta', '\omega'};
%sparamix    = [1 2 5]; % Alpha beta omega locations for subjects
%paramix     = [1 1 1 1 1 1; 2 2 2 2 2 2; 3 4 4 5 0 0];
%paramcorrs  = zeros(nfittrials,length(sparamix),ngroups);

%for i = 1:ngroups
%    for m = 1:length(sparamix)
%        figure();
%        for k = 1:nfittrials
%            [~,bestmodeli] = max(best_pxp(i,:,k));
%            x = twostepfit.trial(k).experiment.groups(i).subjects.params(:,sparamix(m));
%            if m == 3 && bestmodeli == 5
%                y = zeros(size(x));
%            elseif m == 3 && bestmodeli == 6
%                y = ones(size(x));
%            else
%                y = twostepfit.trial(k).experiment.groups(i).rlfits(bestmodeli).fit.params(:,paramix(m, bestmodeli));
%            end

%            paramcorrs(k, m, i) = corr(x, y);

%            subplot(2,5,k);
%            s = pqscatter(x, y, {'match', 'lm'});
%            pqtitle(s, [modelnames{bestmodeli}, ' PXP=', num2str(pxp(i,bestmodeli,k))]);
%            xlabel('Actual'); ylabel('Estimate');

%        end
%        suptitle([sparamnames{m}, ' Parameter Estimates for Group', groupnames{i}]);
%    end
%end

%figure();
%for i = 1:ngroups
%    subplot(3,2,i);
%    s = pqbar(1:nfittrials, paramcorrs(:,:,i));
%    pqtitle(s, ['Fitted Parameter Correlations for Group ', groupnames{i}]);
%end
