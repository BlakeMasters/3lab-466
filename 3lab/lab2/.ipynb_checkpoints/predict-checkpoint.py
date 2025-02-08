import sys
import json
import pandas as pd
from modified_csv import csv_handler
from c45 import c45, c45Node
#Blake Masters
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py <CSVFile> <JSONFile> [eval]")
        sys.exit(1)
        
    csv_filename = sys.argv[1]
    json_filename = sys.argv[2]
    eval_flag = (len(sys.argv) > 3 and sys.argv[3].lower() == "eval")
    df, target, type_dict = csv_handler(csv_filename)
    
    model = c45(split_metric="Gain", threshold=0.1)
    model.read_tree(json_filename)

    predictions = model.predict(df)
    
    if not eval_flag:
        for pred in predictions:
            print(pred)
    else:
        true_labels = df[target].tolist()
        for i, pred in enumerate(predictions):
            print(f"Row {i+1}: Prediction = {pred}, True = {true_labels[i]}")
        total = len(true_labels)
        correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
        incorrect = total - correct
        accuracy = correct / total
        print("\nEvaluation:")
        print(f"Total records classified: {total}")
        print(f"Correctly classified: {correct}")
        print(f"Incorrectly classified: {incorrect}")
        print(f"Overall accuracy: {accuracy:.4f}")
        classes = sorted(set(true_labels))
        matrix = {cl: {c: 0 for c in classes} for cl in classes}
        for t, p in zip(true_labels, predictions):
            matrix[t][p] += 1
        print("Confusion Matrix:")
        for cl in classes:
            print(f"{cl}: {matrix[cl]}")

if __name__ == "__main__":
    main()