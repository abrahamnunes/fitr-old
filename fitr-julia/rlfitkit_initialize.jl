using Distributions,
      StatsBase,
      DataFrames,
      GLM,
      Optim,
      PyPlot

include("Utilities/utilities.jl")
include("Types/types.jl")
include("Types/typemethods.jl")
include("Fitting/optimizer.jl")
include("Fitting/LikelihoodFunctions/gonogobandit_lik.jl")
include("Tasks/gonogobandit.jl")
include("Visualization/visfunctions.jl")

#=

 GENERATE DATA

=#

subjects   = _newsubjects("Group 1",
                          100,
                          ["Learning Rate", "Inverse Temperature"],
                          [Uniform(0, 1), Uniform(2, 8)])

simopts    = SimOptions(true, "results/figures/exp1", "svg")

experiment = _gonogobandit(201, 0.5, subjects, simopts)

#=

 SPECIFY MODELS

=#

model = RLModel("RW Model",
                ["Learning Rate", "Inverse Temperature"],
                ["unit", "pos"],
                zeros(experiment.subjects.N, 2))

#==============================================================

 MODEL FITTING

==============================================================#
fitopts = FitOptions(2, 10)

model = _optimize(experiment, model, fitopts)

#==============================================================

 PLOT RESULTS

==============================================================#

figure()
subplot(1, 2, 1)
plot(linspace(0, 1, 100), linspace(0, 1, 100), lw = 2, color = "black")
scatter(experiment.subjects.params[:,1], model.params[:,1], c = "yellow", alpha = 0.5)
xlim(0, 1)
ylim(0, 1)
title(L"\alpha")

subplot(1, 2, 2)
scatter(experiment.subjects.params[:,2], model.params[:,2], c="red", alpha = 0.5)
title(L"\beta")
