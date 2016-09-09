#===============================================================================

  PLOTCUMSUMREWARD Plots the cumulative sum of rewards for each agent as a line

    INPUTS:
      experiment = structure with task, results, and subject data
      alpha      = [0, 1] transparency of lines

===============================================================================#

function _plotcumsumreward(experiment, alpha, plotdir, filetype)
  rewards = experiment["results"]["rewards"]

  ntrials = size(rewards, 1)
  trials  = linspace(1, ntrials, ntrials)

  figure()
  plot(trials, cumsum(rewards, 1), alpha = alpha)
  xlim(1, ntrials)
  title("Cumulative Rewards by Subject over Trials\n")
  xlabel("\nTrial")
  ylabel("Cumulative Reward\n")

  savefig(string(plotdir, "/cumulativerewardplot.", filetype))
end

#===============================================================================

  REWARDPARAMSUBPLOT Creates set of subplots with parameters correlated against
   total reward and against eachother

    INPUTS:
      experiment = structure with task, results, and subject data
      alpha    = [0, 1] transparency of lines

===============================================================================#

function _rewardparamsubplot(experiment::RLExperiment, alpha::Float64, plotdir::ASCIIString, filetype::ASCIIString)
  nparams = size(experiment.subjects.params, 2)
  rewards = experiment.rewards
  params  = experiment.subjects.params

  idx     = reshape(1:(nparams+1)^2, nparams+1, nparams+1)'
  idxdiag = diag(idx)
  histidx = idxdiag[2:end]

  xmargin = 0.4;
  figure(figsize = (10, 10))

  for i = 1:nparams+1
    for j = i:nparams+1
      subplot(nparams+1, nparams+1, idx[i, j])
      if idx[i, j] == 1
        ntrials = size(rewards, 1)
        trials  = linspace(1, ntrials, ntrials)
        plot(trials, cumsum(rewards, 1), alpha = alpha)
        xlim(1, ntrials)
        tight_layout()
        title("Reward")
        xlabel("Trial")
        ylabel("Total Reward")
      elseif in(idx[i, j], histidx)
        minx = minimum(params[:,i-1])
        maxx = maximum(params[:,i-1])

        plt[:hist](params[:,i-1], Int64(round(0.2*size(params[:,i-1], 1))))
        xlim(minx - xmargin*(maxx-minx), maxx + xmargin*(maxx-minx))
        title(string(experiment.subjects.paramnames[j-1], " Distribution"))
        tight_layout()

      elseif i == 1 && j > 1
        df     = DataFrame()
        df[:x] = params[:,j-1]
        df[:y] = vec(sum(rewards, 1)')
        lfit   = fit(LinearModel, y ~ x, df)
        slope  = coef(lfit)[2]
        intcpt = coef(lfit)[1]

        if confint(lfit)[2,1] < 0 && 0 < confint(lfit)[2,2]
          clr = "blue"
        else
          clr = "red"
        end

        scatter(df[:x], df[:y], c = clr, alpha = alpha)
        minx = minimum(df[:x])
        maxx = maximum(df[:x])
        g = linspace(minx - xmargin*(maxx-minx), maxx + xmargin*(maxx-minx), 100)
        plot(g, slope*g + intcpt, lw=2, color = "black")
        xlim(minx - xmargin*(maxx-minx), maxx + xmargin*(maxx-minx))
        text(minx - (xmargin-0.1)*(maxx-minx), maximum(df[:y]), string(L"r = ", round(cor(df[:x], df[:y]), 2)))
        tight_layout()
        title(string(experiment.subjects.paramnames[j-1], " .vs Reward" ))
      else
        df     = DataFrame()
        df[:x] = params[:,j-1]
        df[:y] = params[:,i-1]
        lfit   = fit(LinearModel, y ~ x, df)
        slope  = coef(lfit)[2]
        intcpt = coef(lfit)[1]

        if confint(lfit)[2,1] < 0 && 0 < confint(lfit)[2,2]
          clr = "blue"
        else
          clr = "red"
        end

        scatter(df[:x], df[:y], alpha = alpha)

        minx = minimum(df[:x])
        maxx = maximum(df[:x])
        g = linspace(minx - xmargin*(maxx-minx), maxx + xmargin*(maxx-minx), 100)
        plot(g, slope*g + intcpt, lw=2, color = "black")
        xlim(minx - xmargin*(maxx-minx), maxx + xmargin*(maxx-minx))
        text(minx - (xmargin-0.1)*(maxx-minx), maximum(df[:y]), string(L"r = ", round(cor(df[:x], df[:y]), 2)))
        tight_layout()
        title(string(experiment.subjects.paramnames[j-1], " .vs ", experiment.subjects.paramnames[i-1]))
      end
    end
  end

  savefig(string(plotdir, "/rewardparameterplot.", filetype))

end

#===============================================================================

  REWARDRASTER Creates a reward raster plot with subjects on y axis and trials
    on the x axis

===============================================================================#

function _rewardraster(experiment::RLExperiment, plotdir::ASCIIString, filetype::ASCIIString)
    rewards   = experiment.rewards
    ntrials   = size(rewards, 1)
    nsubjects = experiment.subjects.N

    figure()
    imshow(rewards, extent = [1, ntrials, 1, nsubjects])
    title("Rewards Received Over Trials")
    xlabel("Trial")
    ylabel("Subject")

    savefig(string(plotdir, "/rewardrasterplot.", filetype))

end
