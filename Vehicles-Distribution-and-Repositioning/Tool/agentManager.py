import numpy as np

class AgentArray(object):
    def __init__(self, data=None):
        self.data = data or []

    def __setitem__(self, index, value): # change value
        if index >= (self.data):  # The assignment is that the given index is out of range, then add value after the list
            self.data.append(value)
            return
        self.data[index] = value

    def __getitem__(self, index):  # get value
        return self.data[index]

    def getId(self,id):
        for i in range(len(self.data)):
            if self.data[i].id == id:
                return self.data[i]
        raise Exception("错误id")

    def append(self,value):
        self.data.append(value)
        return

    def getMaxDS(self):  # get value
        result = 0
        for i in range(len(self.data)):
            temp = abs(len(self.data[i].demandBuffer) - self.data[i].supply)
            if temp > result:
                result = temp
        return result

    def getDSMeanAndStd(self):  # get value
        x_mean = 0
        x_std = 0
        arrayDemand = list()
        for i in range(len(self.data)):
            arrayDemand.append((len(self.data[i].demandBuffer) - self.data[i].supply))
        x_mean = sum(arrayDemand) / len(arrayDemand)
        x_std = np.std(arrayDemand)
        return x_mean, x_std

    def getMaxDemand(self):  # get value
        result = 0
        for i in range(len(self.data)):
            temp = len(self.data[i].demandBuffer)
            if temp > result:
                result = temp
        return result

    def getDemandMeanAndStd(self):  # get value
        x_mean = 0
        x_std  = 0
        arrayDemand = list()
        for i in range(len(self.data)):
            arrayDemand.append(len(self.data[i].demandBuffer))
        x_mean = sum(arrayDemand)/len(arrayDemand)
        x_std  = np.std(arrayDemand)
       
        return (x_mean,x_std)

    def getMaxSupply(self):  # get value
        result = 0
        for i in range(len(self.data)):
            temp = self.data[i].supply
            if temp > result:
                result = temp
        return result

    def getSupplyMeanAndStd(self):  # get value
        x_mean = 0
        x_std  = 0
        arrayDemand = list()
        for i in range(len(self.data)):
            arrayDemand.append(self.data[i].supply)
        x_mean = sum(arrayDemand)/len(arrayDemand)
        x_std  = np.std(arrayDemand)
        
        return x_mean,x_std