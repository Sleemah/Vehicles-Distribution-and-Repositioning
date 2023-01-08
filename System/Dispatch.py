from ortools.graph import pywrapgraph
from Tool import ToolFunction
from Algorithm.KMAlgorithm import run_kuhn_munkres
from config import configParaser

class Dispatch(object):
    def __init__(self):
        pass

    #Output an n*4 array, the first column is the origin area, the second column is the destination area, and the third column is the number of repositions
    #The 4th column is its Manhattan distance
    def KMDispath(self,repositionArray,HuntArray):
        tempResult = []
        result = []
        for i in range(len(repositionArray)):
            for j in range(len(HuntArray)):
                temp = ToolFunction.getManDist(repositionArray[i].id,HuntArray[j].id)
                if(temp<=ToolFunction.getMaxDist()):
                    tempWeight = -temp*abs(
                        repositionArray[i].repositionV - HuntArray[j].huntV
                    )
                    tempResult.append((i,j,tempWeight))
        tempResult = run_kuhn_munkres(tempResult)
        for i in range(len(tempResult)):
            tempDist = ToolFunction.getManDist(repositionArray[tempResult[i][0]].id,HuntArray[tempResult[i][1]].id)
            if tempDist > ToolFunction.getMaxDist():
                continue
            result.append([repositionArray[tempResult[i][0]].id,HuntArray[tempResult[i][1]].id,
                           int(repositionArray[tempResult[i][0]].repositionV),tempDist*configParaser.timeScale])
        return result

    def NFDispath(self, repositionArray, HuntArray):
        result = []
        bias = len(repositionArray)
        total = bias + len(HuntArray)
        totalRep = 0
        totalHun = 0
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(repositionArray)):
            for j in range(len(HuntArray)):
                temp = ToolFunction.getManDist(repositionArray[i].id, HuntArray[j].id)
                if (temp <= ToolFunction.getMaxDist()):
                    tempWeight = temp * abs(
                        repositionArray[i].repositionV - HuntArray[j].huntV
                    )
                    #tempWeight = temp
                    min_cost_flow.AddArcWithCapacityAndUnitCost(i+1, int(j + bias + 1),
                                                                int(repositionArray[i].repositionV),
                                                                int(tempWeight))
        for i in range(len(repositionArray)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(0, i+1,
                                                        int(repositionArray[i].repositionV),
                                                        0)
            min_cost_flow.SetNodeSupply(i+1, 0)
            totalRep += int(repositionArray[i].repositionV)
        for i in range(len(HuntArray)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(int(i + bias + 1), total + 1,
                                                        int(HuntArray[i].huntV),
                                                        0)
            min_cost_flow.SetNodeSupply(i+bias+1, 0)
            totalHun += int(HuntArray[i].huntV)
        totalRep = min(totalRep,totalHun)
        min_cost_flow.SetNodeSupply(0, int(totalRep))
        min_cost_flow.SetNodeSupply(total+1, -int(totalRep))
        if min_cost_flow.SolveMaxFlowWithMinCost() ==  min_cost_flow.OPTIMAL:
            for i in range(min_cost_flow.NumArcs()):
                if min_cost_flow.Tail(i) != 0 and min_cost_flow.Head(i) != total + 1:
                    tempDist = ToolFunction.getManDist(repositionArray[min_cost_flow.Tail(i) - 1].id,
                                                       HuntArray[min_cost_flow.Head(i) - bias - 1].id)
                    if tempDist > ToolFunction.getMaxDist() and min_cost_flow.Flow(i) <= 0:
                        continue
                    result.append(
                        [repositionArray[min_cost_flow.Tail(i) - 1].id, HuntArray[min_cost_flow.Head(i) - bias - 1].id,
                         min_cost_flow.Flow(i), tempDist * configParaser.timeScale])
        else:
            print('There was an issue with the min cost flow input.')

        return result
