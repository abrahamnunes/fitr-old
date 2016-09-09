#===============================================================================

  GONOGOBANDIT_LIK Computes the log-likelihood for the Go-Nogo Bandit task

===============================================================================#

function _gonogobandit_lik(params,
                           states,
                           actions,
                           rewards)

   #Initialize variables
   ntrials = length(states)
   Q       = zeros(2, 2)
   alpha   = params[1]
   beta    = params[2]
   loglik  = 0

   for t = 1:ntrials
       s = states[t]
       a = actions[t]
       r = rewards[t]

       # Compute log-likelihood
       loglik = loglik + beta*Q[s, a] - _logsumexp(beta*Q[s, :])

       # Learn
       Q[s, a] = Q[s, a] + alpha*(r - Q[s, a])

   end

   return loglik
end
