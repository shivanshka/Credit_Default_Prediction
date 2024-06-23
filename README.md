# Give-Me-Some-Credit-Machine-Learning

## Problem Statement

Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit.

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this project is to build a model utilizing the historical data on 250K
borrowers to predict if someone/borrower will experience financial distress in the next 2 years. This is a supervised machine learning problem, specifically binary classificaton problem.

## Dataset Description

The dataset has been taken from the [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) which contains Behavioural and Demographic Information of the borrowers like Age, Debt Ratio, Number of Dependents in the family etc..
The training dataset contains 150k customer information, while test dataset contains
100k customer information.

## Components

The package is comprised of several modules:

1. Data Ingestion: This module is responsible to download the data from the [GitHub repository](https://github.com/Chirag1994/Datasets) and extracts the necessary files from it. For this project we have used only the training dataset since we have complete information on that dataset. After extracting the zip file, we split the downloaded dataset into training and validation datasets.

2. Data Validation: This module is responsible for checking if the dataset follows the dataset schema i.e., if the features are of specific data type as when the model was trained on.

3. Data Transformation: This module is responsible for applying all the feature-engineering steps for model building and transforms the traininig as well as the validation datasets. The feature-engineering steps are all saved in a pickle object for making all the preprocessing steps and transforming the new unseen data.

4. Model Trainer: This module is responsible for training the model on the transformed training dataset (achieved in Data Transformation stage). As part of the training, we trained Logistic Regression, Random Forest and XGBoost models.

## Pipeline

The pipeline package contains 2 modules:

1. Training_pipeline: This module is responsible for instantiating the entire pipeline starting right from Data Ingestion till Model Training.

2. Prediction_pipeline: This module is responsible for getting the predictions on new unseen data. This supports both single as well as batch prediction. The predictions are made using the model saved after training the model into a pickle object.

### Tech Stack

- Programming Language: Python3
- Data Manipulation and Analysis: Pandas, Matplotlib, Seaborn
- Model Training: Scikit-Learn
- Containerizing the application: Docker
- Web Development: HTML, CSS, Flask
- CI/CD Workflow: GitHub Actions
- Deployment: AWS ECR and AWS EC2
