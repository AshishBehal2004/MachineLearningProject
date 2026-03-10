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

    def initiate_model_trainer(self, train_arr, test_arr):
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

            model_report: dict= evaluate_models(X_train = X_train,
                                                y_train = y_train,
                                                X_test = X_test,
                                                y_test  = y_test,
                                                models=models )
            
            #best model score from dict
            best_model_score = 0
            for model_name in model_report:
                score = model_report[model_name]["roc_auc "]
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = model_name

            # extracting bes model name from the index at w hich best_model_score is found, so the value(best_mode_score)
            # will give its key(best_model_name), doing this using list 
            
            best_model = models[best_model_name] 

            if best_model_score < 0.6 :
                raise CustomException("No best model found!!")

            logging.info("Best found model on both training and test dataset")

            save_object( #dumping the best model found using model.pkl above as its name
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            print("Best model f1: " , f1_score(y_test, predicted))

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)