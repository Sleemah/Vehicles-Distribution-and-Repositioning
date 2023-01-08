import pandas as pd
import numpy as np

class CSVTool(object):
    def __init__(self,savePath):
        self.fp = open(savePath, "w+", encoding="utf-8")
        self.savePath = savePath
        return

    def saveFile(self, my_list):
        """
         Save the file as a file in csv format, write the column name in the header, and write the row name in the index
        :param my_list: a list data to be stored
         :return:
         """
        count = 1
        self.fp = open(self.savePath, "a+", encoding="utf-8")
        for value in my_list:
            self.fp.write(str(value))
            if count != len(my_list):
                self.fp.write(',')
            count += 1
        self.fp.write('\n')
        self.fp.close()

    def readFile(self,str):
        data= pd.read_csv(str)
        train_data = np.array(data)  # np.ndarray()
        train_x_list = train_data.tolist()  # list
        return train_x_list

    def __del__(self):
        if self.fp != None:
            self.fp.close()