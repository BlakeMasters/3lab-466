#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from modified_csv import csv_handler  # your CSV handler module
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(data, types, target):
    """
    Preprocess the DataFrame by encoding categorical variables.
    - data: the pandas DataFrame returned by csv_handler.
    - types: a dictionary mapping column names to types ("numeric", "ignore", or a string for categorical).
    - target: the target variable column name.
    Returns: (X, y) where X is the feature DataFrame and y is the target Series.
    """
    # Drop the target column from features
    X = data.drop(columns=[target])
    y = data[target]

    for col in X.columns:
        col_type = types.get(col, "numeric")
        if col_type != "numeric":
            X[col] = X[col].astype(str)
            encoder = OrdinalEncoder()
            X[[col]] = encoder.fit_transform(X[[col]])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X, y


def main():
    if len(sys.argv) != 3:
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

    if mode == "sklearn":
        print("Using scikit-learn RandomForestClassifier...")

        X, y = preprocess_data(data, types, target)

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define a hyperparameter grid for the Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30]
        }

        # Initialize the RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)

        # Use GridSearchCV to perform hyperparameter tuning with 5-fold cross-validation
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_

        # Evaluate on the training set
        y_pred_train = best_rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        # Evaluate on the test set
        y_pred_test = best_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Generate the confusion matrix for the test set
        conf_matrix = confusion_matrix(y_test, y_pred_test)

        # Print out the results
        print("Best Hyperparameters:")
        print(grid_search.best_params_)
        print("\nTraining Accuracy: {:.2f}%".format(train_accuracy * 100))
        print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
        print("\nConfusion Matrix (Test):")
        print(conf_matrix)
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred_test))
    else:
        print("Using custom RandomForest implementation (466 version)...")


if __name__ == "__main__":
    main()

    """Your evaluation program shall report both training and test evaluation results. 
    The following shall be included in the output: • Selected Method: make sure your output unambiguously identifies whether the results were obtained from the sklearn implementation or from your implementation. • Grid search values for hyperparameters that were tested. Output the range of hyperparamter values that were used for the evaluation. You should also output the splitting metric and the splitting threshold that you pass through to C45 (even if those values are fixed- the goal here is to provide a comprehensive output). • Best hyperparameter values. Report the hyperparameter values that resulted in the best predictions. • Training accuracy and Confusion Matrix. Report the training accuracy, and the training confusion matrix for the best model. • Test accuracy and test Confusion Matrix. Report the test accuracy and the test confusion matrix for the best model. • Precision/Recall/F1. For two-class classification problems only together with training and test accuracy, report the training and test precision, recall, and f-measure (F1) for each of the two classes."""