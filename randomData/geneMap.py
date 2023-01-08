from config import configParaser
import random
import numpy as np
from Tool import dataSave
from pandas.core.frame import DataFrame
from Tool import csvIO

mapPath =configParaser.mapPath
mapPath = '../'+ mapPath

csvTool = csvIO.CSVTool(mapPath)
height = configParaser.mapHeight
width = configParaser.mapWidth
csvTool.saveFile([width,height])
count = 1

for i in range(height):
    tempResult = []
    for j in range(width):
        tempResult.append(count)
        count += 1
    csvTool.saveFile(tempResult)