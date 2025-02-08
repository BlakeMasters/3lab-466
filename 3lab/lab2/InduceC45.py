import sys
import json
import numpy as np
import pandas as pd
import sklearn as sk
from collections import defaultdict
from modified_csv import csv_handler
from c45 import c45, c45Node
#Blake Masters
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 InduceC45 <TrainingSetFile.csv> [<fileToSave>]")
        sys.exit(1)
    csv_filename = sys.argv[1]

    df, target, type_dict = csv_handler(csv_filename)
    c45_instance = c45(split_metric="Gain", threshold=0.1, attribute_types=type_dict)
    if len(sys.argv) > 2:
        save_filename = sys.argv[2]
        saving = True
        data_name = csv_filename
    else:
        saving = False
        save_filename= None
        data_name = None
        
    c45_instance.fit(df, target, save=saving, output_filename=save_filename, dataset_filename=data_name)
    
    if data_name == None:
        print("\nOutput JSON representation of the induced decision tree:")
        print(c45_instance.to_output_json(dataset_filename=csv_filename))