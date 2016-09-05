'''
    FIT.py
        Functions to fit models to data
'''

import scipy.optimize as op

def fitr(model, data, maxiters=1000, nstarts=2):
    '''
        Fits reinforcement learning model to behavioural data
    '''

    likfx     = model['likfx']
    nparams   = len(model['params'])
    nsubjects = len(data)
    ntrials   = np.size(data[0]['results']['s'], axis=0)

    # Extract parameter range specifications from the model dict
    paramrng = []
    for key in model['params']:
        paramrng.append(model['params'][key]['rng'])

    # Initialize the output structure
    fit = {'nsubjects'  : nsubjects,
           'ntrials'    : ntrials,
           'nparams'    : nparams,
           'params'     : np.zeros([nsubjects, nparams]),
           'loglik'     : np.zeros([nsubjects, 1]),
           'logpost'    : np.zeros([nsubjects, 1])}

    # Loop over iterations, subjects, and parameter initializations
    for k in range(maxiters):
        for s in range(0, nsubjects):
            print('Fitting subject ' + str(s) + '...')
            data_s = data[s]['results']
            logpostfx = lambda params: -likfx(paramtransform(params, paramrng, 'uc'), data_s)
            for i in range(nstarts):
                initparams = rnd.normal(0, 1, nparams)
                opres = op.minimize(logpostfx, initparams, method='Nelder-Mead', options={'disp': False})
                if i == 0 or fit['logpost'] < -opres['fun']:
                    fit['logpost'] = -opres['fun']
                    fit['loglik']  = likfx(paramtransform(opres['x'], paramrng, 'uc'), data_s)
                    fit['params'][s, :] = opres['x']

    # Convert estimated parameters to constrained space
    for s in range(0, nsubjects):
        fit['params'][s,:] = paramtransform(fit['params'][s,:], paramrng, 'uc')

    return fit
