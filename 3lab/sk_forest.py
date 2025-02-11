#!/usr/bin/env python3
# !/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from modified_csv import csv_handler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(data, types, target):
    # Drop the target column from features
    X = data.drop(columns=[target])
    y = data[target]

    # For each column in X, if its type is not "numeric", then treat it as categorical.
    for col in X.columns:
        col_type = types.get(col, "numeric")
        if col_type != "numeric":
            # Convert to string first (in case there are numeric-like values)
            X[col] = X[col].astype(str)
            encoder = OrdinalEncoder()
            X[[col]] = encoder.fit_transform(X[[col]])
        else:
            # Ensure numeric columns are of numeric type
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X, y


def main():
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python3 rfEval.py <CSV filename> <Method>")
        sys.exit(1)

    csv_filename = sys.argv[1]
    mode = sys.argv[2].lower()

    print(f"Selected Mode: {mode}")

    if not os.path.exists(csv_filename):
        print(f"Error: File '{csv_filename}' does not exist.")
        sys.exit(1)

    try:
        data, target, types = csv_handler(csv_filename)
    except Exception as e:
        cwd = os.getcwd()
        raise ValueError(f"Input: {csv_filename} is not a valid file path from {cwd} to a modified csv") from e

    # Branch based on the selected mode
    if mode == "sklearn":
        X, y = preprocess_data(data, types, target)

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30]
        }
        print("\nGrid Search Parameter Grid:")
        print(param_grid)

        random_forest = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_

        # Evaluate on the training set
        y_pred_train = best_rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_conf_matrix = confusion_matrix(y_train, y_pred_train)

        # Evaluate on the test set
        y_pred_test = best_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_conf_matrix = confusion_matrix(y_test, y_pred_test)

        # Print out the results
        print("\nBest Hyperparameters:")
        print(grid_search.best_params_)
        print("\nTraining Accuracy: {:.2f}%".format(train_accuracy * 100))
        print("Training Confusion Matrix:")
        print(train_conf_matrix)
        print("\nTest Accuracy: {:.2f}%".format(test_accuracy * 100))
        print("Test Confusion Matrix:")
        print(test_conf_matrix)

        # Print classification reports for both training and test sets
        print("\nClassification Report (Training):")
        print(classification_report(y_train, y_pred_train))
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred_test))

    else:
        print("Using custom RandomForest implementation (466 version)...")


if __name__ == "__main__":
    # Redirect all output to a text file.
    output_filename = "sk_forest_output.txt"
    with open(output_filename, "w") as out_file:
        # Redirect stdout so all print() calls write to out_file
        sys.stdout = out_file
        main()

