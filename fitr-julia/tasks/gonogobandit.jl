function _gonogobandit(ntrials::Int, ptrans::Float64, subjects::SubjectGroup, simopts::SimOptions)

    # Set up simulation parameters
    nsubjects = subjects.N
    alpha     = subjects.params[:, 1]
    beta      = subjects.params[:, 2]
    preward   = [0.6 0.1; 0.1 0.6]

    # Set up variables
    states  = fill(0, ntrials, nsubjects)
    actions = fill(0, ntrials, nsubjects)
    rewards = fill(0, ntrials, nsubjects)


    for i = 1:nsubjects
        Q       = zeros(2, 2)
        for t = 1:ntrials

            # Select state
            s             = rand(Binomial(1, ptrans), 1)[1] + 1
            states[t, i]  = s

            # Select action
            a             = maximum(rand(Multinomial(1, vec(_softmax(beta[i].*Q[s,:]))), 1).*[1; 2])
            actions[t, i] = a

            # Return reward
            r = rand(Binomial(1, preward[s, a]))
            rewards[t,i] = r

            # Learn
            Q[s, a] = (1-alpha[i])*Q[s, a]
            Q[s, a] = Q[s, a] + alpha[i]*(r - Q[s, a])

        end
    end

    experiment = RLExperiment("Go No-Go Bandit",
                              ntrials,
                              subjects,
                              states,
                              actions,
                              rewards)

    if simopts.plotresults == true
        _rewardparamsubplot(experiment, 0.5, simopts.plotfolder, simopts.plottype)
        _rewardraster(experiment, simopts.plotfolder, simopts.plottype)
    end

    println("------ COMPLETED SIMULATED EXPERIMENT ------")
    return experiment

end
