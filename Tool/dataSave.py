import numpy as np
import pandas as pd

def SaveData(data,output):
    col1 = data.values[:, 0]
    col2 = data.values[:, 1]
    col3 = data.values[:, 2]
    col4 = data.values[:, 3]

    data_arry = []
    data_arry.append(col1)
    data_arry.append(col2)
    data_arry.append(col3)
    data_arry.append(col4)
    np_data = np.array(data_arry)

    np_data = np_data.T
    np.array(np_data)
    save = pd.DataFrame(np_data, columns=['time','origin','destination','value'])
    save.to_csv(output, index=False)          # index=False, header=False means not to save the row index and column header
