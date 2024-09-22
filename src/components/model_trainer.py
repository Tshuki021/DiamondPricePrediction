## Basic Import
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException  # Custom exception handler
from src.logger import logging  # Custom logging module
from src.utils import save_object, evaluate_model  # Utility functions for model evaluation and saving
from dataclasses import dataclass  # For creating configuration classes

# Define the configuration for model training using a dataclass
@dataclass
class ModelTrainerConfig:
    """
    ModelTrainerConfig class defines the file path for saving the trained model.
    The trained model will be saved as 'model.pkl' under the 'artifacts' directory.
    """
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# Create a class for Model Training
class ModelTrainer:
    """
    ModelTrainer class handles the process of training multiple regression models, 
    evaluating their performance, and saving the best-performing model.
    """
    def __init__(self):
        # Initialize the ModelTrainerConfig to access the model file path
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Initiates the model training process which involves:
        1. Splitting the training and testing data into features (X) and target (y).
        2. Training multiple regression models.
        3. Evaluating each model using a predefined evaluation function.
        4. Identifying the best model based on R-squared score.
        5. Saving the best model as a pickle file.

        Args:
            train_array (numpy array): Training data containing features and target.
            test_array (numpy array): Test data containing features and target.
        """
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            # Split the train and test data into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last one for features
                train_array[:, -1],  # Last column for the target variable (y)
                test_array[:, :-1],  # All columns except the last one for test features
                test_array[:, -1]  # Last column for the test target variable (y)
            )

            # Dictionary of models to train and evaluate
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor()
            }

            # Evaluate each model and return a report with the R-squared scores
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the highest model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Find the name of the best-performing model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print('\n================================================================================\n')
            logging.info(f'Best Model Found, Model name : {best_model_name}, R2 Score : {best_model_score}')

            # Save the best-performing model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            # Log and raise a custom exception in case of any errors during model training
            logging.info('Exception occurred during Model Training')
            raise CustomException(e, sys)
