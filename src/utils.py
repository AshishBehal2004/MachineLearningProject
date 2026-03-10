import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import f1_score, roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)   


def evaluate_models(X_train, y_train, X_test, y_test, models ):
    report = {} #dictionary for storing each model's score
    try:
        for i in range(len(list(models))): #looping over every model in model dict.
            model = list(models.values())[i] # picks the actual model object at ith index

            model.fit(X_train, y_train) # Training the model

            y_train_pred = model.predict(X_train) # makes predictions on training data

            y_test_pred = model.predict(X_test) # make predictions on test data

            train_f1= f1_score(y_train,y_train_pred)# f1: balances between precision and recall on training data
            train_roc_auc_score= roc_auc_score(y_train,y_train_pred) # auc: how well model separates fraud vs non-fraud on training data
                    
            test_f1 = f1_score(y_test, y_test_pred) # f1: balances between precision and recall on test data
            test_roc_auc_score = roc_auc_score(y_test, y_test_pred) # auc: how well model separates fraud vs non-fraud on test data

            report[list(models.keys())[i]] = { # saving the model's name and its testscore into the report dict.
                "f1 " : test_f1,
                "roc_auc " :test_roc_auc_score }
            
        return report 
    except Exception as e:
        raise CustomException(e, sys)
    