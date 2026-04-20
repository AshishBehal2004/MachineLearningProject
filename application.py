from flask import Flask, request,render_template
import numpy as mp
import pandas as pd
from src.pipeline.predict_pipeline import CustomData , PredictPipeline
from sklearn.preprocessing import StandardScaler 

application= Flask(__name__) # creates flask app
app= application

# real rows from dataset
FRAUD_ROW = [406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474, 0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919, 0.0]

NORMAL_ROW = [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]


#Route for Homepage

@app.route('/') # when user visits localhost:5000/

def index():
    return render_template("index.html") #send index.html to browser

@app.route('/predictdata', methods=['GET', 'POST']) # this route handles both page load and button click

def predict_datapoint():
    if request.method == 'GET': # when user just visits the webpage
        return render_template('home.html') #shows the page with buttons
    
    else:
        transaction_type= request.form.get("transaction_type") # when user clicks a button

        if transaction_type == "fraud": # identifies which button was clicked
            row = FRAUD_ROW
        else:
            row= NORMAL_ROW
        
        data= CustomData(row=row) # then wraps the row in CustomData class
        pred_df= data.get_data_as_dataframe() # then converts to DataFrame

        pipeline= PredictPipeline() # then creates pipleine object
        result = pipeline.predict(pred_df) # scale + r

        label= "Fraud Detected" if result[0] == 1 else "Legitimate Transaction"
        return render_template('home.html', results=label)
    
if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=5001) #starts the server