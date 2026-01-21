import os
import sys # need to use custom exception hence the reason of the import
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass # using to automatically create boilerplate code (like __init__, __repr__, __eq__) 
# for simple data-holding classes in order to define classes cleanly without writing repetitive code.

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered The data ingestion method or component")

        try:
            df= pd.read_csv('notebook/data/creditcard.csv') #reading the data set
            logging.info('Read the dataset as the dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #converting the raw data into csv file

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42) #training the data

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()