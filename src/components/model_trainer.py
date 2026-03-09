import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier #build trees sequentially, each tree corrects the mistake of previous one to detect rare fraud cases
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Splitting trainnig and test input data")
            X_train, y_train, X_test, y_test= (
                train_arr[:, :-1], # picks all rows and columns except last 
                train_arr[:,-1], # picks all rows and only the last column
                test_arr[:,:-1], # picks all rows and columns except last 
                test_arr[:,-1] # picks all rows and only the last column
            )

            models = {
                "LogisticRegression" : LogisticRegression(),
                "RandomForestClassifier" : RandomForestClassifier(),
                "GradientBoostingClassifier" : GradientBoostingClassifier(),
            }

            model_report: dict= evaluate_models(X=X_train,
                                                y= y_train,
                                                x_test= X_test,
                                                y_test= y_test,
                                                models=models )
        except:
            pass