from Algorithm import vpg
import numpy as np
import random
from config import configParaser

class VPGschedule(object):
    def __init__(self,id,a_dim,s_dim,a_bound = 1):
        vpg.seed_torch(configParaser.randomSeed)
        self.ddpg = vpg.VPG(a_dim,s_dim,a_bound)
        self.a_bound = a_bound
        self.var = 3  # control exploration
        self.MEMORY_CAPACITY = configParaser.MEMORY_CAPACITY
        self.id = id
        self.v = 0
        self.logp = 0

    def getAction(self,state, DSMap = None):
        action,self.v,self.logp = self.ddpg.choose_action(state)
        return action

    def storeTransition(self,currentTransition):         #0 current state, 1 current action, 2 reward, 3 future state
        self.ddpg.buf.store(currentTransition[0], currentTransition[1], currentTransition[2],self.v, self.logp)

    def processNextState(self,currentTransition):
        self.storeTransition(currentTransition)

    def learn(self):
        self.ddpg.buf.finish_path(self.v)
        loss_a, loss_c = self.ddpg.learn()
        return loss_a, loss_c


def _init(agentNum,a_dim = 1,a_bound = 1):
    global ddpgSchedule
    s_dim = int(agentNum + configParaser.timeNum + 1)
    ddpgSchedule = VPGschedule(a_dim,s_dim,a_bound )

def getddpg():
    return ddpgSchedule