# Credit Card Fraud Detection Project (EDA Phase)

## Project Overview
This project focuses on performing Exploratory Data Analysis (EDA) on a credit card transactions dataset to understand patterns between fraudulent and non-fraudulent transactions.
The goal of this phase is to clean the data, explore important features, and gain insights that will later support machine learning model development.

## Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (or specify your source)
- **Size**: 284,807 transactions
- **Features**: 30 features (V1-V28 are PCA-transformed, Time, Amount, Class)
- **Class Distribution**: Highly imbalanced (~0.17% fraud cases)

## Project Goals
- Understand the structure and characteristics of credit card transaction data
- Identify patterns and anomalies in fraudulent vs. legitimate transactions
- Handle missing values and data quality issues
- Analyze class imbalance and its implications
- Prepare data for future machine learning model development

## Technologies Used
- **Python 3.13.9**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Data preprocessing and future modeling

## Key Analysis Performed

### 1. Data Loading and Initial Exploration
- Loaded the dataset and examined basic statistics
- Checked data types, missing values, and dataset dimensions
- Analyzed class distribution (fraud vs. non-fraud)

### 2. Feature Analysis
- **Transaction Amount**: 
  - Compared distribution between fraud and legitimate transactions
  - Identified that fraudulent transactions tend to have different amount patterns
  
- **Temporal Patterns**:
  - Analyzed transaction frequency over time
  - Explored fraud occurrence patterns across different time periods
  
- **PCA Features (V1-V28)**:
  - Examined distributions and correlations
  - Identified which features show significant differences between classes

### 3. Class Imbalance Analysis
- Visualized the severe class imbalance (0.17% fraud)
- Calculated imbalance ratio
- Planned strategies for handling imbalance in future modeling:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Undersampling majority class
  - Class weight adjustment
  - Ensemble methods

### 4. Data Preprocessing
- Handled missing values (if any)
- Feature scaling for Amount and Time features
- Split data into training and testing sets
- Prepared data pipeline for model training

## Key Findings
1. **Transaction Amount**: Fraudulent transactions have a different distribution pattern compared to legitimate ones
2. **Temporal Patterns**: Fraud occurrence shows specific time-based patterns
3. **Class Imbalance**: Severe imbalance requires special handling techniques
4. **Feature Importance**: Certain PCA features show strong correlation with fraud



## Contact
**Ashish Behal**
- GitHub: [@AshishBehal2004](https://github.com/AshishBehal2004)
- LinkedIn: [Ashish Behal](https://linkedin.com/in/ashishbehal)
- Email: ashishbehal52@gmail.com