# Importing necessary modules and components
import os
import sys
from src.logger import logging  # Custom logging module to track logs
from src.exception import CustomException  # Custom exception handler for error handling
import pandas as pd  # Library for data manipulation

# Importing the DataIngestion, DataTransformation, and ModelTrainer classes from their respective components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Main execution block to initiate the data ingestion, transformation, and model training process
if __name__ == '__main__':
    try:
        # Step 1: Instantiate the DataIngestion object
        obj = DataIngestion()

        # Step 2: Start the data ingestion process to load and split the dataset
        # The method returns the file paths for the train and test datasets
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        # Print the paths of the ingested train and test datasets for verification
        print(train_data_path, test_data_path)

        # Step 3: Instantiate the DataTransformation object
        data_transformation = DataTransformation()

        # Apply data transformation on the train and test datasets
        # This method returns transformed numpy arrays for train and test data
        train_arr, test_arr, _ = data_transformation.initiate_data_transormation(train_data_path, test_data_path)

        # Step 4: Instantiate the ModelTrainer object
        model_trainer = ModelTrainer()

        # Initiate the model training process using the transformed train and test arrays
        model_trainer.initiate_model_training(train_arr, test_arr)
    
    except Exception as e:
        # Handle any unexpected exceptions during the process by logging them and raising a custom exception
        logging.error('Error occurred during the execution of the pipeline.')
        raise CustomException(e, sys)
