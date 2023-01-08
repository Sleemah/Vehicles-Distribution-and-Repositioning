import sys
import subprocess
import pkg_resources
from sklearn.linear_model import LinearRegression 
import lightgbm as ltb
import math
import pandas as pd
import numpy as np
import seaborn as sns

class Ph1(object):
     
    data = pd.read_csv("C:\\Users\\slema\\Downloads\\E2.csv") 
    data.head()

    #x = data.iloc[:,1:2].values
    #y= data.iloc[:,2].values

    x = np.array(data[["#Orders"]])
    y = np.array(data["#Vehicles"])
    
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, 
                                                    y,train_size=0.8, test_size=0.2, 
                                                    random_state=42)  

    def linearRegression(self,X_train, X_test, Y_train, Y_test):  
        model = ltb.LGBMRegressor()
        model.fit(X_train, Y_train)   
        ypred = model.predict(X_test)
        data = pd.DataFrame(data={"Predicted Vehicle": ypred.flatten()})
        print(data.head())
        score = model.score(X_test, Y_test) 
        return model 
