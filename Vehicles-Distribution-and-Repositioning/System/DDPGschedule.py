from Algorithm import ddpg
from Algorithm import c_ddpg
import numpy as np
import random
from config import configParaser

ddpgSchedule = None

class DDPGschedule(object):
    def __init__(self,id,a_dim,s_dim,a_bound = 1):
        ddpg.seed_torch(configParaser.randomSeed)
        if configParaser.method in [1,5]:
            self.ddpg = ddpg.DDPG(a_dim,s_dim,a_bound)
        if configParaser.method == 2:
            self.ddpg = c_ddpg.DDPG(a_dim,s_dim,a_bound)
        self.a_bound = a_bound
        self.var = 3  # control exploration
        self.MEMORY_CAPACITY = configParaser.MEMORY_CAPACITY
        self.id = id

    def getAction(self,state, DSMap = None):
        action = self.ddpg.choose_action(state)
        #if self.ddpg.pointer < self.MEMORY_CAPACITY:
        #    action = random.uniform(-1,1)
        #else:
        action = np.clip(np.random.normal(action.cpu(), self.var), -self.a_bound, self.a_bound)  # add randomness to action selection for exploration
        return action

    def storeTransition(self,currentTransition):            #0 current state, 1 current action, 2 DSMap, 3 reward, 4 future state
        if configParaser.method == 1:
            self.ddpg.store_transition(currentTransition[0], currentTransition[1],
                                       currentTransition[2], currentTransition[3])
        if configParaser.method == 2:
            self.ddpg.store_transition(currentTransition[0], currentTransition[1],
                                        currentTransition[2], currentTransition[3],
                                       currentTransition[4], currentTransition[5])

    def processNextState(self,currentTransition):
        self.storeTransition(currentTransition)

    def learn(self):
        if self.ddpg.pointer > self.MEMORY_CAPACITY:
            self.var *= .9995  # decay the action randomness
            temp_lossa = 0
            temp_lossc = 0
            for j in range(configParaser.LEARN_TIMES):
                loss_a, loss_c = self.ddpg.learn()
                temp_lossa += loss_a / configParaser.LEARN_TIMES
                temp_lossc += loss_c / configParaser.LEARN_TIMES
            return temp_lossa,temp_lossc
        return 0,0

def _init(a_dim = 1,a_bound = 1):
    global ddpgSchedule
    if ddpgSchedule is None:
        s_dim = int(configParaser.agentNum + configParaser.timeNum + 1)
        ddpgSchedule = DDPGschedule(0,a_dim,s_dim)

def getddpg():
    return ddpgSchedule