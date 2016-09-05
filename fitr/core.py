'''
    CORE FUNCTIONS
'''

def runparadigm(params, nsubjects=50, paradigm=bandit):
    '''
        Runs simulated subjects (defined by `params` dictionary) and runs them through a simulated task specified by `paradigm` 
    '''
    data = {}
    for i in range(0, nsubjects):
        results = paradigm(lr=params['lr'][i], itemp=params['itemp'][i])
        data[i] = {'results': results}
    return data
