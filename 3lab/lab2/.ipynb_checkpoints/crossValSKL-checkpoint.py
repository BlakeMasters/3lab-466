import sys
import json
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from modified_csv import csv_handler
#Blake Masters
def compute_confusion_matrix(true_labels, predictions, classes):
    matrix = {cl: {c: 0 for c in classes} for cl in classes}
    for t, p in zip(true_labels, predictions):
        matrix[t][p] += 1
    return matrix

def cross_validate_sklearn(df, target_col, hyperparams, type_dict, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    results = []
    classes = sorted(y.unique())
    numeric_cols = [col for col, t in type_dict.items() if t == "numeric" and col != target_col]
    categorical_cols = [col for col, t in type_dict.items() if t != "numeric" and t != "ignore" and col != target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ]
    )

    thresholds = hyperparams.get("InfoGain", [])
    for thresh in thresholds:
        all_true = []
        all_pred = []
        for train_idx, test_idx in kf.split(X):
            X_train_raw = X.iloc[train_idx].copy()
            X_test_raw = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("clf", DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=0,
                    min_impurity_decrease=thresh))
            ])
            pipeline.fit(X_train_raw, y_train)
            preds = pipeline.predict(X_test_raw)
            all_true.extend(y_test.tolist())
            all_pred.extend(preds.tolist())
        acc = accuracy_score(all_true, all_pred)
        cm = compute_confusion_matrix(all_true, all_pred, classes)
        results.append({
            "split_metric": "Gain (sklearn: 'entropy')",
            "threshold": thresh,
            "accuracy": acc,
            "confusion_matrix": cm
        })
    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python crossValSKL.py <CSV filename> <Hyperparameter grid JSON filename> [output_tree_image]")
        sys.exit(1)
    csv_filename = sys.argv[1]
    grid_filename = sys.argv[2]
    output_tree_file = sys.argv[3] if len(sys.argv) > 3 else None

    df, target, type_dict = csv_handler(csv_filename)
    for col, t in type_dict.items():
        if t == "numeric":
            df[col] = df[col].astype(float)
    target_col = target.strip().lower()
    print("Using target column:", target_col)

    with open(grid_filename, "r") as f:
        hyperparams = json.load(f)
    if "Ratio" in hyperparams:
        print("Warning: Ignoring Ratio hyperparameters; using InfoGain only.")

    cv_results = cross_validate_sklearn(df, target_col, hyperparams, type_dict, n_folds=10)
    best_config = max(cv_results, key=lambda x: x["accuracy"])
    print("\nBest Model Hyperparameters (Scikit-Learn):")
    print(f"Splitting Metric: {best_config['split_metric']}")
    print(f"Threshold (min_impurity_decrease): {best_config['threshold']}")
    print(f"Overall CV Accuracy: {best_config['accuracy']:.4f}")
    print("Confusion Matrix:")
    for cl in sorted(df[target_col].unique()):
        print(f"{cl}: {best_config['confusion_matrix'][cl]}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', [col for col, t in type_dict.items() if t == "numeric" and col != target_col]),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [col for col, t in type_dict.items() if t != "numeric" and t != "ignore" and col != target_col])
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(
            criterion="entropy",
            random_state=0,
            min_impurity_decrease=best_config["threshold"]))
    ])
    pipeline.fit(X, y)
    print("\nFinal model trained on the entire dataset.")

    if output_tree_file:
        plt.figure(figsize=(20,10))
        plot_tree(pipeline.named_steps["clf"],
                  feature_names=X.columns,
                  class_names=[str(cl) for cl in sorted(np.unique(y))],
                  filled=True, rounded=True)
        plt.savefig(output_tree_file)
        print(f"Decision tree visualization saved to {output_tree_file}")

if __name__ == "__main__":
    main()