from Algorithm import dqn
import numpy as np
import random
from config import configParaser

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.5
EPSILON_FINAL = 0.02
discreteDict={0:0.9,1:0.6,2:0.3,3:0,4:-0.3,5:-0.6,6:-0.9}
#discreteDict={0:0.9,1:0.3,2:0,3:-0.3,4:-0.9}
class DQNschedule(object):
    def __init__(self,id,a_dim,s_dim,a_bound = len(discreteDict)):
        dqn.seed_torch(configParaser.randomSeed)
        self.ddpg = dqn.DQN(a_dim,s_dim,a_bound)
        self.a_bound = a_bound
        self.MEMORY_CAPACITY = configParaser.MEMORY_CAPACITY
        self.id = id

    def getAction(self,state, DSMap = None):
        if np.random.random() < self.ddpg .epsilon:
            a = random.randint(0,self.a_bound-1)
        else:
            a = self.ddpg .choose_action(state)
        return discreteDict[a]

    def decayEpsilon(self,count):
        self.ddpg.epsilon = max(EPSILON_FINAL, EPSILON_START - count / EPSILON_DECAY_LAST_FRAME)

    def storeTransition(self,currentTransition):           #0 current state, 1 current action, 2 DSMap, 3 reward, 4 future state
        currentTransition[1] = list(discreteDict.keys())[list(discreteDict.values()).index(currentTransition[1])]
        self.ddpg.store_transition(currentTransition[0], currentTransition[1],
                                    currentTransition[2], currentTransition[3])

    def processNextState(self,currentTransition):
        self.storeTransition(currentTransition)

    def learn(self):
        if self.ddpg.pointer > self.MEMORY_CAPACITY:
            temp_lossa = 0
            for j in range(configParaser.LEARN_TIMES):
                loss_a = self.ddpg.learn()
                temp_lossa += loss_a / configParaser.LEARN_TIMES
            return temp_lossa
        return 0

def _init(agentNum,a_dim = 1,a_bound = 7):
    global ddpgSchedule
    s_dim = int(agentNum + configParaser.timeNum + 1)
    ddpgSchedule = DQNschedule(a_dim,s_dim,a_bound )

def getddpg():
    return ddpgSchedule