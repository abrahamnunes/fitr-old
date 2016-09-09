#===============================================================================

 SIMULATIONOPTIONS TYPE

===============================================================================#

type SimOptions
    plotresults::Bool
    plotfolder::ASCIIString
    plottype::ASCIIString
end

#===============================================================================

 SUBJECTGROUP
  - Declares type
  - Specifies method to greate a subject group

===============================================================================#

type SubjectGroup
    name::ASCIIString
    N::Int64
    paramnames::Array{ASCIIString, 1}
    paramdist
    params::Array{Float64, 2}
end

#===============================================================================

 RLEXPERIMENT TYPE

===============================================================================#

type RLExperiment
    name::ASCIIString
    ntrials::Int
    subjects::SubjectGroup
    states::Array
    actions::Array
    rewards::Array
end

#===============================================================================

 MODEL TYPE

===============================================================================#

type RLModel
    name::ASCIIString
    paramnames::Array{ASCIIString, 1}
    paramrng::Array{ASCIIString, 1}
    params::Array{Float64, 2}
end

#===============================================================================

 MODEL FITTING OPTIONS

===============================================================================#

type FitOptions
    iterations::Int
    nstarts::Int
end
