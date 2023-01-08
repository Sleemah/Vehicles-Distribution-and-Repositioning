



def rewardFunction(reward,takeAction = False):
    bias = 0
    result = 0
    if takeAction is False:
        bias = 0             #0.5 reward value brought by 20% difference between supply and demand
    result = reward
    #result = 1-pow(reward ,0.4) -  bias
    result = 1 - reward - bias
    return result