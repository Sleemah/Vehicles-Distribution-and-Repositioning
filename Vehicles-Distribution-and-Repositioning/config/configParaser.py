import configparser
from Tool import csvIO
import random
import os

count = 0

randomSeed= 0
mapPath = ''
savePath = ''
savePath_loss = ''
fleetSize = 0
dataPath= ''
actionThreshold = 0
maxDist = 0
lifeTime = 0
maxEpoch = 0
timeInterval = 0
method = 0
methodDict = {'random': 0, 'ddpg': 1,"cddpg" : 2,"dqn" : 3,"vpg":4,"ddpg_s":5}
timeNum = 0
timeBegin = 0
timeEnd = 0
timeDay = 0
mapHeight = 0
mapWidth  = 0
agentNum = 0
timeScale = 0
processesNum = 0

MAX_EPISODES = 0
MAX_EP_STEPS = 0
LR_A = 0
LR_C = 0
GAMMA = 0
TAU = 0
MEMORY_CAPACITY = 0
BATCH_SIZE = 0
LEARN_TIMES = 0

dataSize = 0

def _init():
    global randomSeed, mapPath, savePath, fleetSize, dataPath, actionThreshold, maxDist, lifeTime, maxEpoch, method,timeInterval,timeNum,timeScale
    global timeBegin,timeEnd,timeDay,savePath_loss,mapHeight,mapWidth,agentNum,processesNum
    global MAX_EPISODES, MAX_EP_STEPS, LR_A, LR_C, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, LEARN_TIMES
    global dataSize
    config = configparser.ConfigParser()
    root_path = os.path.abspath(os.path.dirname(__file__))#.split('CitySimulator')[0]           #linux Path
    config.read(root_path + '/config.ini', encoding='utf-8')
    #root_path = os.path.abspath(os.path.dirname(__file__))                                     #Win Path
    #config.read(root_path+'\\config.ini', encoding='utf-8')
    randomSeed = config.getint('config', 'randomSeed')
    mapPath = config.get('config', 'mapPath')
    savePath = config.get('config', 'savePath')
    savePath_loss = config.get('config', 'savePath_loss')
    fleetSize = config.getint('config', 'fleetSize')
    mapHeight = config.getint('config', 'mapHeight')
    mapWidth = config.getint('config', 'mapWidth')
    #agentNum = mapHeight*mapWidth
    dataPath = config.get('config', 'dataPath')
    actionThreshold = config.getfloat('config', 'threshold')
    maxDist = config.getint('config', 'maxDist')
    lifeTime = config.getint('config', 'lifeTime')
    timeInterval = config.getint('config', 'timeInterval')
    maxEpoch = config.getint('config', 'epoch')
    method = methodDict[config.get('config', 'method')]
    timeBegin = config.getint('config', 'timeBegin')
    timeEnd = config.getint('config', 'timeEnd')
    timeDay = config.getint('config', 'timeDay')
    #timeNum = int(24*(60/timeInterval))
    timeNum = timeEnd - timeBegin
    timeScale = config.getfloat('config','timeScale')
    processesNum = config.getint('config','processes')

    MAX_EPISODES = config.getint('DDPG', 'MAX_EPISODES')
    MAX_EP_STEPS = config.getint('DDPG', 'MAX_EP_STEPS')
    LR_A = config.getfloat('DDPG', 'LR_A')  # learning rate for actor
    LR_C = config.getfloat('DDPG', 'LR_C')  # learning rate for critic
    GAMMA = config.getfloat('DDPG', 'GAMMA') # reward discount
    TAU = config.getfloat('DDPG', 'TAU')  # soft replacement
    MEMORY_CAPACITY = config.getint('DDPG', 'MEMORY_CAPACITY')
    BATCH_SIZE = config.getint('DDPG', 'BATCH_SIZE')
    LEARN_TIMES = config.getint('DDPG', 'LEARN_TIMES')

    dataSize = config.getint('randomData', 'dataSize')



if __name__ == '__main__':
    print('cannot be executed as main program')
else:
    if count == 0:
        _init()
        print('Initialization parameters complete')
        count += 1