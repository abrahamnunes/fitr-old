#===============================================================================

 LOGSUMEXP

  Avoids numerical overflow/underflow

===============================================================================#

function _logsumexp(x)

  ym = maximum(x)
  yc = x - ym
  y  = ym + log(sum(exp(yc)))
  i  = find(!isfinite(ym))
  if !isempty(i)
    y[i] = ym[i]
  end

  return y

end

#===============================================================================

 SOFTMAX FUNCTION

  Computes softmax probabilities for an input vector `x`.

===============================================================================#

function _softmax(x)
  return exp(x)./sum(exp(x))
end

#===============================================================================

 VARIABLE TRANSFORMATION

===============================================================================#

function _transformvar(x, transtype::ASCIIString, rng::Array{ASCIIString,1})
    nvars = length(x)
    for i = 1:nvars
      if transtype == "uc"
          if rng[i] == "unit"
              x[i] = 1./(1+exp(- maximum([x[i], -10.0])))
          elseif rng[i] == "pos"
              x[i] = exp(minimum([x[i], 10.0]))
          end
      elseif transtype == "cu"
          if rng[i] == "unit"
              x[i] = log(x[i]./(1-x[i]))
          elseif rng[i] == "pos"
              x[i] = log(x[i])
          end
      end
  end
  return x
end
