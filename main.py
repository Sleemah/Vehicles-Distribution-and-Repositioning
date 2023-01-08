
from config import configParaser
from System.Dispatch import Dispatch
import numpy as np
import random
import configparser
import math
import copy
from Agent.Region import Region
from Agent.Demand import Demand
from Tool import csvIO
from System.Reposition import Reposition
from System.Hunt import Hunt
from Tool.agentManager import AgentArray
from Tool import ToolFunction
from System import DDPGschedule
from Tool.MyThread import MyThread
from Tool.Phase1 import Ph1



csvTool = csvIO.CSVTool(configParaser.savePath)
csvToolLoss = csvIO.CSVTool(configParaser.savePath_loss)
dispatchMoudule = Dispatch()
#############################use variables#########################################
agentNum = 0                                        #number of areas
cityMap = []                                        #city ​​map
agentArray = AgentArray()                           #Area management array
randomSeed = -1                                     #The random seed to use (negative means the seed is not fixed)
fleetSize = 0                                      #Group size
demandArray = []                                   #array of requirements
maxTime = 0                                        #Simulated maximum time
actionThreshold = 0                                #action threshold
maxDist = 2                                        #maxDist = 2 #The maximum allowed Reposition distance
lifeTime = 2                                       #order lifetime
maxEpoch = 999                                     #Maximum number of rounds
####################ADDING PHASE ONE ###########################################################
ph1 = Ph1()
#############################read parameters##########################################

randomSeed = configParaser.randomSeed
mapPath = configParaser.mapPath
savePath = configParaser.savePath
fleetSize = configParaser.fleetSize
dataPath = configParaser.dataPath
actionThreshold = configParaser.actionThreshold
maxDist = configParaser.maxDist
lifeTime = configParaser.lifeTime
maxEpoch = configParaser.maxEpoch
method = configParaser.method
if randomSeed>-1 :
    random.seed(randomSeed)

###############################################################################

#############################read map##########################################
print('start reading map...')
cityMap = np.loadtxt(mapPath, delimiter=',',skiprows = 1,dtype = int)
[rows, cols] = cityMap.shape
for i in range(rows):
    for j in range(cols):
        if cityMap[i][j] > 0 :
            agentNum += 1
configParaser.agentNum = agentNum
tempCount = 0
for i in range(rows):
    for j in range(cols):
        if cityMap[i][j] > 0 :
            x =Region(cityMap[i][j], tempCount,maxDist)
            agentArray.append(x)
            tempCount += 1
ToolFunction._init(cityMap,maxDist,agentNum)                         #Helper class

###############################################################################

#############################read data##########################################
print('Start reading data...')
temp = np.loadtxt(dataPath, delimiter=',',skiprows = 1,dtype = float)
[rows, cols] = temp.shape
for i in range(rows):
    if temp[i][0] < configParaser.timeBegin or temp[i][0] > configParaser.timeEnd:
        continue
    demandArray.append(Demand(temp[i][0],temp[i][1],temp[i][2],temp[i][3], lifeTime))
demandArray.sort()
maxTime = temp[i][0]
###############################################################################
                                  ###############################initialization##########################################
print('Initialize number of vehicle...')

    
modell = ph1.linearRegression(ph1.xtrain, ph1.xtest, ph1.ytrain, ph1.ytest )

                

                ###############################################################################  


for epoch in range(maxEpoch):
    print('Processing round %d...' % (epoch+1),end='')
    currentTime = configParaser.timeBegin
    currentDemand = 0
    cnnFeature = []
    huntAgent = []
    repositionAgent = []
    dispathchRes = []
    supplyArray = []
    totalReposition = 0
    totalKMReposition1 = 0
    totalKMReposition2 = 0
    tempMaxDS = 1000
    tempReward = 0
    tempDecayCount =  0
    
    while currentTime <  configParaser.timeEnd:
        huntAgent.clear()
        repositionAgent.clear()
        threads = []
        while demandArray[currentDemand].time == currentTime:  # Count the generated passengers in the current time period
            agentArray.getId(demandArray[currentDemand].origin).demand += 1
            agentArray.getId(demandArray[currentDemand].origin).demandBuffer.append(demandArray[currentDemand])
            currentDemand += 1
            
            x = math.ceil(modell.predict([[ agentArray.getId(demandArray[currentDemand].origin).demand ]]) )
            
            
            if currentDemand >= len(demandArray):
                break
            
            if ( x > fleetSize):
                 x  = fleetSize        #when the model prediction exceed the total fleet size, will have the same size
            for i in range(agentNum):
                agentArray[i].supply  =x
                agentArray[i].supply_backup =x

        print("after x")
        print(agentArray.getId(demandArray[currentDemand].origin).demand)
                
        while len(supplyArray) > 0:  # Statistics of vehicles arriving in the current time period
            if supplyArray[0].arriveTime > currentTime:
                break
            agentArray.getId(supplyArray[0].destination).supply += 1 #number of V=supply
            supplyArray.pop(0)

        tempMaxDS = agentArray.getDemandMeanAndStd()
        tempDSMap = None
        if method == 2:
            tempDSMap = ToolFunction.calculDSMap(agentArray)
        for i in range(agentNum):  # Update reposition vehicle status
            tempReward += agentArray[i].nextTime(currentTime,tempMaxDS,tempDSMap)
        for i in range(agentNum):  # Perform match operation
            agentArray[i].matchDS(supplyArray, currentTime)
            if method == 3:
                agentArray[i].ddpg.decayEpsilon(tempDecayCount)

        tempMaxDS = agentArray.getDemandMeanAndStd()
        tempDSMap = None
        if method == 2:
            tempDSMap = ToolFunction.calculDSMap(agentArray)

        for i in range(agentNum):
            thread = MyThread(agentArray[i].schedule,args=(currentTime,tempMaxDS,tempDSMap))
            threads.append(thread)
            thread.start()
        i = 0
        for t in threads:
            t.join()
            tempAction = t.get_result()
            if tempAction >= actionThreshold and agentArray[i].supply > 0:  # Action grouping
                repositionAgent.append(Reposition(agentArray[i].id, tempAction, agentArray[i].supply))
            if tempAction <= -actionThreshold:
                huntAgent.append(Hunt(agentArray[i].id, tempAction, agentArray[i].supply))
            agentArray[i].ignoreAction = True   #Always add actions to the replay buffe
            i = i+1
        totalKMReposition1 =  totalKMReposition1+len(repositionAgent)
        totalKMReposition2 =  totalKMReposition2+len(huntAgent)
        dispathchRes = dispatchMoudule.NFDispath(repositionAgent, huntAgent)  # Perform Dispatch operation
        totalReposition += len(dispathchRes)
        for i in range(len(dispathchRes)):  # Perform reposition operation
            agentArray.getId(dispathchRes[i][0]).repositionVehicle(dispathchRes[i][2])
            agentArray.getId(dispathchRes[i][1]).acceptVehicle(supplyArray, currentTime,dispathchRes[i][3], dispathchRes[i][2],dispathchRes[i][0])
        supplyArray.sort()
        currentTime += 1
        tempDecayCount += 1

    if method in [1,2,4]:  #DDPG learning & VPG learning
        temp_lossa, temp_lossc = 0,0
        for i in range(configParaser.agentNum):
            temp1, temp2 = agentArray[i].ddpg .learn()
            temp_lossa += temp1 / configParaser.agentNum
            temp_lossc += temp2 / configParaser.agentNum
        tempSave = [temp_lossa,temp_lossc]
        csvToolLoss.saveFile(tempSave)
        print(' actor loss is %.4f , critic loss is %.4f' % (temp_lossa, temp_lossc), end='')
    elif method == 3:
        temp_lossa = 0
        for i in range(configParaser.agentNum):
            temp1 = agentArray[i].ddpg.learn()
            temp_lossa += temp1 / configParaser.agentNum
        tempSave = [temp_lossa]
        csvToolLoss.saveFile(tempSave)
        print(' q loss is %.4f' % (temp_lossa), end='')

    epoch += 1
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(agentNum):
        sum1 += agentArray[i].matchOrder
        sum2 += agentArray[i].unmatchOrder
        sum3 += agentArray[i].imbalanceNum
        agentArray[i].reInitial()
    tempSave = [sum1,sum2,sum3, '%.2f%%' % (sum1 / (sum1 + sum2) * 100),tempReward]
    csvTool.saveFile(tempSave)
    print(' average reward %.4f' % (tempReward / agentNum), end='')
    print(' Number of Reposition areas %d %d %d' % (totalKMReposition1,totalKMReposition2,totalReposition), end='')
    print(' Average Disequilibrium Supply and Demand Quantity %.2f' % (sum3/(agentNum)),end='')
    print(' match order %d, lost order %d' % (sum1, sum2),end='')
    print(' match rate %.2f%%' % (sum1 / (sum1 + sum2) * 100))
    
    
    
#    agentArray = copy.deepcopy(agentArrayInit)
    if (epoch+1)%500 == 0:
        for i in range(agentNum):
            agentArray[i].saveModel('checkpoints')





