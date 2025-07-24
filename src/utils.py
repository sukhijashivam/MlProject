import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        # Get the directory path from the full file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and dump the object into it using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # If anything goes wrong, raise a custom exception with the original exception and system info
        raise CustomException(e, sys)
    

from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save only test score (you can also store both if needed)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

