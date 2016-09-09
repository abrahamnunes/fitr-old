#===============================================================================

 NEWSUBJECTS Creates a group of new subjects

===============================================================================#

function _newsubjects(groupname::ASCIIString,
                      N::Int64,
                      paramnames::Array{ASCIIString, 1},
                      paramdist)

    K = length(paramnames)
    params  = zeros(N, K)
    for k = 1:K
        params[:,k] = rand(paramdist[k], N)
    end

    group = SubjectGroup(groupname, N, paramnames, paramdist, params)
    return group
end
