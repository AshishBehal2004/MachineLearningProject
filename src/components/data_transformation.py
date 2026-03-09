import sys 
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass #why data class used here?
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self, numerical_columns): 
        '''This function is responsible got data transformation based on different types of data'''
        try:
            target_column_name = "Class"
            num_columns = numerical_columns

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {num_columns}")

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            target_column_name = "Class"

            numerical_columns = train_df.columns.drop(target_column_name)

            preprocessing_obj = self.get_data_transformer_obj(numerical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
            logging.info(f"Saved preprocessing object")


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
           
        except Exception as e:
            raise CustomException(e, sys)