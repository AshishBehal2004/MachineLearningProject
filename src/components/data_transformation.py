import sys 
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass #why data class used here?
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifcats','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["Time", "Amount"]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")
            preprocessor = ColumnTransformer(
                ("num_pipeline", num_pipeline, numerical_columns)
            )
            return preprocessor
        except:
            pass