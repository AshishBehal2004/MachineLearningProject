import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass
    
    def predict(self, features):
        try: 
            print("loading model")
            model_path= 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path) #loads the saved .pkl file from artifacts folder

            print("model loaded")
            preprocessor = load_object(file_path= preprocessor_path)

            print("preprocessor loaded")
            data_scaled = preprocessor.transform(features) # scale input data same way training data was scaled
            pred_proba = model.predict_proba(data_scaled)
            print("Fraud probability:", pred_proba[:, 1])
            pred = (pred_proba[:, 1] >= 0.2).astype(int)
            print("prediction done")
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, row: list):
        self.row = row

    def get_data_as_dataframe(self):
        columns= ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10', 
                   'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount']
        return pd.DataFrame([self.row],columns=columns) # convert list into dataframe so preprocessor can process it