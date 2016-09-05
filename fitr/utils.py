'''
    Useful functions used across fitr modules
'''

def softmax(x):
    ''' Computes softmax probability '''
    return np.exp(x)/np.sum(np.exp(x))

def mnrandi(p):
    ''' Returns index of max value from a multinomial sample '''
    return np.argmax(rnd.multinomial(1, p))

def logsumexp(x):
    '''
        - Protects against numerical overflow/underflow.

        Based on code from:
            Samuel Gershman's `mfit` package (https://github.com/sjgershm/mfit)
    '''
    ym = np.max(x)
    yc = x - ym
    y  = ym + np.log(np.sum(np.exp(yc)))
    i  = np.argwhere(np.logical_not(np.isfinite(ym)))
    if np.size(i) != 0:
        y[i[0][0]] = ym[i[0][0]]
    return y

def paramtransform(params, paramrng, transformtype):
    '''
        Transforms parameters between constrained and unconstrained spaces

        `params`        = 1-D array of K parameter values
        `paramrng`      = parameter range ('unit', 'pos', or 'unc')
                            'unit' -> interval [0, 1]
                            'pos'  -> interval [0, infinity)
                            'unc'  -> interval (-infinity, +infinity)
        `transformtype` = unconstrained -> constrained   ('uc')
                          constrained   -> unconstrained ('cu')

        Based on code from:
            Akam, T., Costa, R., & Dayan, P. (2015). Simple Plans or        Sophisticated Habits? State, Transition and Learning Interactions in the Two-Step Task. PLoS Computational Biology, 11(12), 1–25.
    '''
    K = np.size(paramrng)
    for k in range(0, K):
        if transformtype == 'uc':
            if paramrng[k] == 'unit':
                if params[k] < -16:
                    params[k] = -16
                params[k] = 1./(1 + np.exp(-params[k]))
            elif paramrng[k] == 'pos':
                if params[k] > 16:
                    params[k] = 16
                params[k] = np.exp(params[k])
            elif paramrng[k] == 'unc':
                params[k] = params[k]
            else:
                raise ValueError(paramrng[k] + ' is not a valid parameter range')
        elif transformtype == 'cu':
            if paramrng[k] == 'unit':
                params[k] = -np.log((1./params[k])-1)
            elif paramrng[k] == 'pos':
                params[k] = np.log(params[k])
            elif paramrng[k] == 'unc':
                params[k] = params[k]
            else:
                raise ValueError(paramrng[k] + ' is not a valid parameter range')
    return params
