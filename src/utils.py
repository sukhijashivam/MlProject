import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = {}
        best_model_overall = None
        best_score = float("-inf")

        for model_name in models:
            model = models[model_name]
            param_grid = params.get(model_name, {})

            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            if test_model_score > best_score:
                best_score = test_model_score
                best_model_overall = best_model

        return report, best_model_overall

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
