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
        - Based on Samuel Gershman's `mfit` Matlab package:
            - https://github.com/sjgershm/mfit
    '''
    ym = np.max(x)
    yc = x - ym
    y  = ym + np.log(np.sum(np.exp(yc)))
    i  = np.argwhere(np.logical_not(np.isfinite(ym)))
    if np.size(i) != 0:
        y[i[0][0]] = ym[i[0][0]]
    return y
