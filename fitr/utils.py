'''
    Useful functions used across fitr modules
'''

def softmax(x):
    ''' Computes softmax probability '''
    return np.exp(x)/np.sum(np.exp(x))

def mnrandi(p):
    ''' Returns index of max value from a multinomial sample '''
    return np.argmax(rnd.multinomial(1, p))
