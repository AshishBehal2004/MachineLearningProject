import os
import sys # need to use custom exception hence the reason of the import
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass # using to automatically create boilerplate code (like __init__, __repr__, __eq__) 
# for simple data-holding classes in order to define classes cleanly without writing repetitive code.

@dataclass
class DataIngestionConfig: # This class holds the folder paths where we will save our data files
    train_data_path: str=os.path.join('artifacts',"train.csv") # where to save training data
    test_data_path: str=os.path.join('artifacts',"test.csv") # where to save test data
    raw_data_path: str=os.path.join('artifacts',"data.csv") # where to save raw or original data

class DataIngestion: # This class is responsible for loading and splitting the data
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # Stores the filepaths (where to save train, test and raw data) so we can use them later

    def initiate_data_ingestion(self):
        logging.info("Entered The data ingestion method or component") # log that we have started data ingestion process

        try:
            df= pd.read_csv('notebook/data/creditcard.csv') #reading the data set
            logging.info('Read the dataset as the dataframe') #logging the message

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # creates the artifacts folder if it doesn't already exist

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # converting the raw data into csv file

            logging.info("Train test split initiated") # logging the message
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42) #training the data

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #Save the training data to CSV file (without row index, with column headers)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) #Save the training data to CSV file (without row index, with column headers)

            logging.info("Ingestion of the data is completed") # logging the message

            return( # Returning the filepaths of the saved test and train dataset
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e: # Throws a Custom error message if anything goes wrong
            raise CustomException(e, sys)
if __name__ == "__main__": # Means run this file directly
    obj=DataIngestion() # creates DataIngestion object
    obj.initiate_data_ingestion() # Starts the DataIngestion process