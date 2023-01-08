import random
import System.DDPGschedule as DDPG
import numpy as np
import math
from Agent.Supply import Supply
from config import configParaser
from System.DDPGschedule import DDPGschedule
from System.DQNschedule import DQNschedule
#from System.VPGschedule import VPGschedule
from Tool import ToolFunction
from Algorithm.rewardFun import rewardFunction
import pandas as pd
import numpy as np

class Region(object):
    

    
    def __init__(self, id, agentId,maxDist ):
        
        self.id = id
        self.agentId = agentId
        self.demand = 0
        self.supply = 0               #supply is 
        self.supply_backup = 0
        self.dist = maxDist
        self.demandBuffer = []
        self.vehicleBuffer = []
        self.takeAction = False
        self.ignoreAction = False

        self.matchOrder = 0                                                              #Number of matching orders
        self.unmatchOrder = 0                                                            #Unmatched orders
        self.repositionVehicleNum = 0                                                    #The number of vehicles for reposition
        self.acceptVehicleNum = 0                                                        #The number of vehicles that accept reposition
        self.imbalanceNum = 0                                                            #unbalanced supply and demand quantity
        if configParaser.method in [1,2]:
            self.ddpg = DDPGschedule(self.id,a_dim = 1,s_dim = int(configParaser.agentNum + configParaser.timeNum + 1))
        elif configParaser.method == 3:
            self.ddpg = DQNschedule(self.id, a_dim=1, s_dim=int(configParaser.agentNum + configParaser.timeNum + 1))
        elif configParaser.method == 4:
            pass#self.ddpg = VPGschedule(self.id,a_dim = 1,s_dim = int(configParaser.agentNum + configParaser.timeNum + 1))
        elif configParaser.method == 5:
            DDPG._init()
            self.ddpg = DDPG.getddpg()
        self.currentTransition = []
        for i in range(self.dist):
            self.vehicleBuffer.append(0)
            
    def schedule(self,currentTime,maxDS,DSMap = None):                                             #Make reposition decisions
        self.takeAction = False
        if configParaser.method == 0:
            return random.uniform(-1, 1)
        if configParaser.method in [1,2,3,4,5]:
            currentState = []
            currentState.append(((len(self.demandBuffer)-self.supply)-maxDS[0])/maxDS[1])
            #currentState.append((len(self.demandBuffer) - self.supply))
            temp = ToolFunction.convert_to_one_hot(currentTime-configParaser.timeBegin, configParaser.timeNum)
            for i in range(len(temp)):
                currentState.append(temp[i])
            temp = ToolFunction.convert_to_one_hot(self.agentId)
            for i in range(len(temp)):
                currentState.append(temp[i])
            self.currentTransition.clear()
            tempAction = self.ddpg.getAction(currentState)
            self.currentTransition.append(currentState)
            self.currentTransition.append(tempAction)
            if configParaser.method == 2:
                self.currentTransition.append(DSMap)
            return tempAction
        return random.uniform(-1,1)

    def acceptVehicle(self,supplyArray, currentTime,dist,num,sourceId):
        #self.vehicleBuffer[dist-1] += num
        for i in range(num):
            supplyArray.append(Supply(currentTime, sourceId, self.id, dist))
        self.acceptVehicleNum += num
        self.takeAction = True

    def repositionVehicle(self,num):
        self.supply -= num
        self.repositionVehicleNum += num
        self.takeAction = True

    def nextTime(self,currentTime,maxDS,DSMap = None):                                                     #process reposition
        self.supply += self.vehicleBuffer[0]
        #for i in range(self.dist-1):
        #    self.vehicleBuffer[i] = self.vehicleBuffer[i+1]
        #self.vehicleBuffer[i+1] = 0
        if len(self.demandBuffer) > 0:
            while (currentTime - self.demandBuffer[0].time) >= self.demandBuffer[0].lifeTime: #Overtime order cancellation
                self.demandBuffer.pop(0)
                self.unmatchOrder += 1
                if len(self.demandBuffer) <= 0:
                    break
        ratio = len(self.demandBuffer)
        tempImbalance = len(self.demandBuffer) - self.supply
        if tempImbalance < 0:
            ratio = self.supply
        self.imbalanceNum += abs(tempImbalance)
        if configParaser.method in [1,2,3,4,5]:
            if self.takeAction or self.ignoreAction:
                currentState = []
                if ratio == 0:
                    reward =  1
                else:
                    reward = rewardFunction(abs(tempImbalance)/ratio,self.takeAction)
                #tempValue = 0
                #for i in range(len(self.demandBuffer)):
                #    tempValue += self.demandBuffer[i].value
                #if self.supply == 0:
                #    reward = tempValue
                #else:
                #    reward = tempValue/self.supply
                #reward = reward/100
                #print('reward is %d'%reward)
                currentState.append((tempImbalance-maxDS[0])/maxDS[1])
                #currentState.append(tempImbalance)
                temp = ToolFunction.convert_to_one_hot(currentTime-configParaser.timeBegin, configParaser.timeNum)
                for i in range(len(temp)):
                    currentState.append(temp[i])
                temp = ToolFunction.convert_to_one_hot(self.agentId)
                for i in range(len(temp)):
                    currentState.append(temp[i])
                if len(self.currentTransition)>0:
                    if configParaser.method == 2:
                        self.currentTransition.append(DSMap)
                    self.currentTransition.append(reward)
                    self.currentTransition.append(currentState)
                    self.ddpg.processNextState(self.currentTransition)
                return reward
        return 0


    def matchDS(self,supplyArray,currentTime):                                          #match order
        while(self.supply>0):
            if len(self.demandBuffer) == 0:
                break
            temp = self.demandBuffer.pop(0)
            supplyArray.append(Supply(currentTime,temp.origin,temp.destination,temp.tripTime))
            self.matchOrder += 1
            self.supply -= 1

    def reInitial(self):
        
        self.matchOrder = 0  # Number of matching orders
        self.unmatchOrder = 0  # Unmatched orders
        self.repositionVehicleNum = 0  #The number of vehicles for reposition
        self.acceptVehicleNum = 0  #The number of vehicles that accept reposition
        self.imbalanceNum = 0  # unbalanced supply and demand quantity

        self.demand = 0
        self.supply = self.supply_backup
        self.demandBuffer = []
        self.vehicleBuffer = []
        self.takeAction = False
        self.ignoreAction = False
        self.currentTransition = []
        for i in range(self.dist):
            self.vehicleBuffer.append(0)
        
        
    def saveModel(self, saveModelPath):
        self.ddpg.ddpg.saveModel(saveModelPath,self.id)