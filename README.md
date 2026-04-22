# Credit Card Fraud Detector

<img width="1470" height="956" alt="demo" src="https://github.com/user-attachments/assets/eb1aa59b-d55f-41e2-bcf9-d1af0c51f678" />

A machine learning system that detects fraudulent credit card transactions, deployed as a live web application on AWS.

# About
Credit Card Fraud Detection is a machine learning project that identifies fraudulent credit card transactions from real-world data. The dataset contains 284,807 transactions from European cardholders, of which only 492 are fraud, a 0.17% fraud rate. This extreme class imbalance is the core challenge the project addresses.
 
The model uses a Random Forest classifier with a custom detection threshold of 0.2, optimized using ROC-AUC score rather than accuracy because a model that predicts every transaction as legitimate would be 99.83% accurate and completely useless. The project is structured as a modular ML pipeline with separate components for data ingestion, transformation, and model training.
 
A Python Flask server handles HTTP requests and serves predictions. The application is deployed on AWS Elastic Beanstalk with automated CI/CD via AWS CodePipeline, enabling automatic deployments on every GitHub push.
 
## Table of Contents
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Deployment on AWS Elastic Beanstalk](#deployment-on-aws-elastic-beanstalk)
## Key Features
- **Fraud Detection:** Identifies fraudulent transactions from real credit card data.
- **Imbalanced Classification:** Handles extreme class imbalance (0.17% fraud rate) using custom threshold tuning.
- **Modular Pipeline:** Separate components for data ingestion, transformation, and model training with structured logging and exception handling.
- **Flask Server:** Python Flask server for handling HTTP requests and serving predictions.
- **AWS Deployment:** Deployed on AWS Elastic Beanstalk with CI/CD via CodePipeline for automated deployments.
## Technologies Used
- **Python:** Core programming language for the model and server.
- **AWS Elastic Beanstalk:** Cloud platform for hosting the application.
- **AWS CodePipeline:** CI/CD pipeline for automated deployments from GitHub.
- **Flask:** Python framework for creating the HTTP server.
- **Scikit-learn:** Machine learning library for building and evaluating the model.
- **Numpy & Pandas:** Data manipulation and preprocessing.
- **HTML/CSS:** Frontend development for the web interface.
## Deployment on AWS Elastic Beanstalk
 
1. **Create Elastic Beanstalk Environment:**
   * Use the AWS console to create a new Elastic Beanstalk environment.
   * Select Python as the platform.
   * Ensure your entry point file is named `application.py` with a Flask object named `application`.
2. **Set Up CodePipeline:**
   * Connect your GitHub repository to AWS CodePipeline.
   * Configure the pipeline to trigger on every push to the `main` branch.
   * Set the deploy stage to target your Elastic Beanstalk environment.
3. **requirements.txt:**
   * Ensure all dependencies are listed with pinned versions:
     ```
     scikit-learn==1.7.2
     Flask
     pandas
     numpy
     dill
     ```
 
4. **Deploy:**
   * Push your code to GitHub. CodePipeline automatically picks up the changes and deploys to Elastic Beanstalk.
   * Monitor the deployment in the CodePipeline console.
5. **Verify:**
   * Access the Elastic Beanstalk environment URL in your browser to confirm the application is running.
## Run Locally
 
```bash
git clone https://github.com/AshishBehal2004/MachineLearningProject.git
cd MachineLearningProject
pip install -r requirements.txt
python application.py
```
 
Then open `http://127.0.0.1:5001`
 
## Dataset
 
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) · 284,807 transactions · European cardholders · September 2013

