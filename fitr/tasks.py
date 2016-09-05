'''
    TASKS
'''

def bandit(ntrials=100,lr=0.01,itemp=4,rsens=1):
    ''' 2 state, 2 action Go-Nogo task'''
    rewardarray = np.array([[1, -1],[-1, 1]])
    states   = np.zeros([ntrials, 1])
    actions  = np.zeros([ntrials, 1])
    rewards  = np.zeros([ntrials, 1])
    rpe = np.zeros([ntrials, 1])
    Q = np.zeros([2, 2]) + 0.5
    for t in range(0, ntrials):
        s = rnd.binomial(1, 0.5)
        a = mnrandi(softmax(itemp*Q[s, :]))
        r = rnd.binomial(1, 0.7)*rewardarray[s, a]
        rpe[t]  = rsens*r - Q[s, a]
        Q[s, a] = Q[s, a] + lr*rpe[t]
        states[t]  = s
        actions[t] = a
        rewards[t] = r
    return {'s': states,
            'a': actions,
            'r': rewards,
            'rpe':rpe}
