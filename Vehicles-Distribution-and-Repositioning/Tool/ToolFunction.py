import numpy as np
from config import configParaser
import copy
from Tool.MyThread import MyThread

cityMap = []
cityManhattanMap = {}
cityManhattanLoc = {}
maxDist = 0
agentNum = 0
threads = []

def _init(_cityMap,_dist,_agentNum):
    global cityMap, maxDist, agentNum
    cityMap = _cityMap
    maxDist = _dist
    agentNum = _agentNum
    [rows, cols] = cityMap.shape
    for i in range(rows):
        for j in range(cols):
            if cityMap[i][j]>0:
                temp = {cityMap[i][j]:copy.deepcopy(cityMap)}
                cityManhattanMap.update(temp)
                cityManhattanLoc.update({cityMap[i][j]:(i,j)})
    for key,values in cityManhattanMap.items():
        thread = MyThread(loadManDist,args=(key,values))
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()
    pass

def getManDist(region1, region2):
    global cityMap,cityManhattanMap
    value = cityManhattanMap.get(region1)
    (row,col) = cityManhattanLoc.get(region2)
    return value[row][col]


def loadManDist(key,values):
    global cityMap
    [rows, cols] = cityMap.shape
    for i in range(rows):
        for j in range(cols):
            if values[i][j] > 0:
                values[i][j] = calManDist(key, values[i][j])

def calManDist(region1, region2):
    global cityMap
    [rows, cols] = cityMap.shape
    x = []
    y = []
    for i in range(rows):
        for j in range(cols):
            if cityMap[i][j] == region1:
                x.append(i)
                x.append(j)
    for i in range(rows):
        for j in range(cols):
            if cityMap[i][j] == region2:
                y.append(i)
                y.append(j)
    return sum(map(lambda i, j: abs(i - j), x, y))

def getMaxDist():
    return maxDist

def convert_to_one_hot(y, C = 0 ):
    if C ==0 :
        C = agentNum
    # Set the number of categories
    num_classes = C
    # the integer to be converted
    arr = y
    # Convert an integer to a 10-bit one hot code
    return np.eye(num_classes,dtype= int)[arr]


def calculDSMap(agentArray):
    row = configParaser.mapHeight
    col = configParaser.mapWidth
    demandMap = np.zeros((row,col),dtype=float)
    supplyMap = np.zeros((row,col),dtype=float)

    #maxDemand = 100
    #maxSupply = 1000

    meanDemand,stdDemand = agentArray.getDemandMeanAndStd()
    meanSupply,stdSupply = agentArray.getSupplyMeanAndStd()
    
    #print('max demand is %d, max supply is %d' % (maxDemand,maxSupply))
    for i in range(row):
        for j in range(col):
            demandMap[i][j] = (len(agentArray[i*5+j].demandBuffer)-meanDemand)/stdDemand
            supplyMap[i][j] = (agentArray[i*5+j].supply-meanSupply)/stdSupply
    demandMap = demandMap[np.newaxis,:]
    supplyMap = supplyMap[np.newaxis,:]
    return np.concatenate((demandMap,supplyMap),axis=0)


def Z_Score(arrayNum):
    arr_ = list()
    x_mean = sum(arrayNum)/len(arrayNum)
    x_std = np.std(arrayNum)
    for x in arrayNum:
        arr_.append((x-x_mean)/x_std)
    return arr_

