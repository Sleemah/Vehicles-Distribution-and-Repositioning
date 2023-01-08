from config import configParaser
import configparser
import random
import numpy as np
from Tool import dataSave
from pandas.core.frame import DataFrame

mapPath =configParaser.mapPath
mapPath = '../'+ mapPath

cityMap = np.loadtxt(mapPath, delimiter=',',skiprows = 1,dtype = int)
outputPath = '../data/testdata.csv'
dataSize = configParaser.dataSize
timeNum = configParaser.timeNum

randomSeed = -1
randomSeed = configParaser.randomSeed

if randomSeed>-1 :
    random.seed(randomSeed)

demandArray = []
temp = dataSize - timeNum
for i in range(timeNum):                               #Randomly assign cars to groups
    tempDemandNum = 1
    if i == timeNum-1:
        tempDemandNum += temp
        for j in range(tempDemandNum):
            origin = random.randint(cityMap.min(), cityMap.max())
            destination = random.randint(cityMap.min(), cityMap.max())
            demandArray.append([i, origin, destination, 0])
        break
    temp1 = int(temp / (timeNum - i))
    temp1 = random.randint(0,temp1)
    tempDemandNum += temp1
    for j in range(tempDemandNum):
        origin = random.randint(cityMap.min(),cityMap.max())
        destination = random.randint(cityMap.min(),cityMap.max())
        demandArray.append([i,origin,destination,0])
    temp -= temp1
print(len(demandArray))
dataFrame = DataFrame(demandArray)
dataSave.SaveData(dataFrame,outputPath)



