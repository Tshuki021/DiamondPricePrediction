
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.exception import CustomException  # Custom exception handler
from src.logger import logging  # Custom logging module
from src.utils import save_object  # Utility function for saving the model
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
    ModelTrainer class handles the process of training the GradientBoostingRegressor 
    and saving the model.
    """
    def __init__(self):
        # Initialize the ModelTrainerConfig to access the model file path
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Initiates the model training process which involves:
        1. Splitting the training and testing data into features (X) and target (y).
        2. Training the GradientBoostingRegressor with predefined parameters.
        3. Saving the trained model as a pickle file.

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

            # Instantiate the GradientBoostingRegressor with predefined parameters
            model = GradientBoostingRegressor(
                learning_rate=0.1,
                max_depth=5,
                n_estimators=100
            )

            logging.info('Training the GradientBoostingRegressor model')
            # Train the model on the training data
            model.fit(X_train, y_train)

            # Evaluate the model performance on the test data (optional)
            r2_score = model.score(X_test, y_test)
            print(f'GradientBoostingRegressor R2 Score on Test Data: {r2_score}')
            logging.info(f'GradientBoostingRegressor R2 Score on Test Data: {r2_score}')

            # Save the trained model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            print('\nModel training completed. The model has been saved successfully!')
            logging.info('Model training completed. The model has been saved successfully!')

        except Exception as e:
            # Log and raise a custom exception in case of any errors during model training
            logging.info('Exception occurred during Model Training')
            raise CustomException(e, sys)
