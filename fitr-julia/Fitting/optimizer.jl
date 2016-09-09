#===============================================================================

  OPTIMIZE

===============================================================================#

function _optimize(experiment::RLExperiment, model::RLModel, opts::FitOptions)
    println(string("Fitting Model: ", model.name))

    iterations = opts.iterations
    nstarts    = opts.nstarts
    nsubjects  = experiment.subjects.N
    nparams    = length(model.paramnames)
    fitparams  = zeros(experiment.subjects.N, nparams)
    logpost    = zeros(experiment.subjects.N)

    # Initialize hyperparameters
    ϕμ = zeros(nparams)
    ϕΣ = eye(nparams, nparams)

    for iter = 1:iterations

        for i = 1:nsubjects
            println(string("Iteration ", iter, ": Fitting Subject ", i))

            S = experiment.states[:, i]
            A = experiment.actions[:, i]
            R = experiment.rewards[:, i]

            #optimization function here
            f(theta) = - _gonogobandit_lik(_transformvar(theta, "uc", model.paramrng), S, A, R)

            for j = 1:nstarts

                initparams = rand(Normal(0, 2), nparams)


                result = optimize(f, initparams, SimulatedAnnealing())
                logp  = - Optim.minimum(result)

                if j == 1 || logpost[i] < logp
                    logpost[i]     = logp
                    fitparams[i,:] = Optim.minimizer(result)
                end

            end
        end

    end

    for i = 1:nsubjects
        model.params[i,:] = _transformvar(fitparams[i,:], "uc", model.paramrng)
    end

    println("------ COMPLETED MODEL FITTING ------")

    return model

end
