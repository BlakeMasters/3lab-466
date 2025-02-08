from sk_forest import sk_forest
from modified_csv import csv_handler
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 rfEval.py <CSV filename> <Method>")
        sys.exit(1)
    csv_filename = sys.argv[1]
    try:data, target_name, types = csv_handler(csv_filename)
    except:
        
        cwd = os.getcwd()
        raise ValueError(f"Input: {csv_filename} is not a valid file path from {cwd} to a modified csv")
    mode = sys.argv[2]
    print(f"Selected Mode: {mode}")
    
    if (len(types) - 1) < 11:
        num_attributes = list(range(4) + 1)
    elif (len(types) - 1) < 21:
        num_attributes = list(range(5) + 1)
    else:
        num_attributes = list(range(7) + 1)
    
   #(M(M-1)(M-2)) / 6 num unique attr triplets
    num_trees = [int(num * (num - 1) * (num - 2) / 6) for num in num_attributes]
    #num_trees = [] #idk range yet
    num_data_points = [] #idk range yet
    
    #c45 probably wants Information Gain and a threshold around 0.1 based on my lab2 results, althought those werent full thorough.
    
    if mode == 'sklearn':
        forest = sk_forest()
        
        
    """Your evaluation program shall report both training and test evaluation results. The following shall be included in the output: • Selected Method: make sure your output unambiguously identifies whether the results were obtained from the sklearn implementation or from your implementation. • Grid search values for hyperparameters that were tested. Output the range of hyperparamter values that were used for the evaluation. You should also output the splitting metric and the splitting threshold that you pass through to C45 (even if those values are fixed- the goal here is to provide a comprehensive output). • Best hyperparameter values. Report the hyperparameter values that resulted in the best predictions. • Training accuracy and Confusion Matrix. Report the training accuracy, and the training confusion matrix for the best model. • Test accuracy and test Confusion Matrix. Report the test accuracy and the test confusion matrix for the best model. • Precision/Recall/F1. For two-class classification problems only together with training and test accuracy, report the training and test precision, recall, and f-measure (F1) for each of the two classes."""