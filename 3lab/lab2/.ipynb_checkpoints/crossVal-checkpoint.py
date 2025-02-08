import sys
import json
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import KFold
from collections import defaultdict
from modified_csv import csv_handler
from c45 import c45, c45Node
#Blake Masters
def compute_confusion_matrix(true_labels, predictions, classes):
    matrix = {cl: {c: 0 for c in classes} for cl in classes}
    for t, p in zip(true_labels, predictions):
        matrix[t][p] += 1
    return matrix

def accuracy_score(true_labels, predictions):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    return np.mean(true_labels == predictions)

def cross_validate(df, target, c45_class, hyperparams, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    results = []
    classes = sorted(df[target].unique())
    for thresh in hyperparams.get("InfoGain", []):
        accuracies = []
        all_true = []
        all_pred = []
        for train_index, test_index in kf.split(df):
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()
            model = c45(split_metric="Gain", threshold=thresh, attribute_types=c45_instance.attribute_types)
            model.fit(train_df, target)
            preds = model.predict(test_df)
            all_true.extend(test_df[target].tolist())
            all_pred.extend(preds)
        acc = accuracy_score(all_true, all_pred)
        cm = compute_confusion_matrix(all_true, all_pred, classes)
        results.append({
            "split_metric": "Gain",
            "threshold": thresh,
            "accuracy": acc,
            "confusion_matrix": cm
        })

    for thresh in hyperparams.get("Ratio", []):
        accuracies = []
        all_true = []
        all_pred = []
        for train_index, test_index in kf.split(df):
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()
            model = c45(split_metric="Ratio", threshold=thresh, attribute_types=c45_instance.attribute_types)
            model.fit(train_df, target)
            preds = model.predict(test_df)
            all_true.extend(test_df[target].tolist())
            all_pred.extend(preds)
        acc = accuracy_score(all_true, all_pred)
        cm = compute_confusion_matrix(all_true, all_pred, classes)
        results.append({
            "split_metric": "Ratio",
            "threshold": thresh,
            "accuracy": acc,
            "confusion_matrix": cm
        })
    return results



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crossVal.py <CSV filename> <Hyperparameter grid JSON filename>")
        sys.exit(1)
    csv_filename = sys.argv[1]
    grid_filename = sys.argv[2]
    df, target, type_dict = csv_handler(csv_filename)
    c45_instance = c45(split_metric="Gain", threshold=0.1, attribute_types=type_dict)
    #hyperparams dict{"InfoGain": [values...], "Ratio": [values...]}
    with open(grid_filename, "r") as f:
        hyperparams = json.load(f)
    
    cv_results = cross_validate(df, target, c45_instance, hyperparams, n_folds=10)
    best_config = max(cv_results, key=lambda x: x["accuracy"])
    print("\nBest Model Hyperparameters:")
    print(f"Splitting Metric: {best_config['split_metric']}, Threshold: {best_config['threshold']}")
    print(f"Overall Cross-Validation Accuracy: {best_config['accuracy']:.4f}")
    print("Confusion Matrix:")
    for true_label in best_config["confusion_matrix"]:
        print(f"{true_label}: {best_config['confusion_matrix'][true_label]}")