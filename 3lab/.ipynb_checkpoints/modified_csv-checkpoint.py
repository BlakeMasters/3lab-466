import csv
import json
import pandas as pd
import numpy as np
#Blake Masters
def csv_handler(filepath):
    with open(filepath, mode ='r')as f:
        data = f.readlines()
        varnames = data[0].strip().split(",")
        num_categories = [int(i) for i in data[1].strip().split(",")]
        target = data[2].strip()
        data_test = np.asarray([row.strip().split(",") for row in data[3:] if row.strip() != ""])
        test = pd.DataFrame(data=data_test, columns = varnames)
        #print(test)
        #print(target)
    type2 = {}
    for i in range(len(num_categories)):
        if num_categories[i] == -1:
            type2[varnames[i]] = "ignore"
        elif num_categories[i] == 0:
            type2[varnames[i]] = "numeric"
        elif num_categories[i] > 0:
            type2[varnames[i]] = str(num_categories[i])
        else:
            raise ValueError(f"num_categories[{i}] = {num_categories[i]} is not in a legal range")
    #print(type2)
    return test,target,type2