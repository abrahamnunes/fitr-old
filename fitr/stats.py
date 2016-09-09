'''
    STATS.py
        fitr Statistical functions
'''

def bic(K, T, loglik):
    ''' Bayesian information criterion '''
    return K*np.log(T) - 2*loglik

def aic(K, loglik):
    ''' Aikake information criterion '''
    return K*2 - 2*loglik

def lme(K, H, logpost):
    ''' Log model-evidence '''
    return logpost + K/2 * np.log(2*np.pi) - np.log(np.linalg.det(H))/2
