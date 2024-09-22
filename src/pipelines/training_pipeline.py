# Importing necessary modules and components
import os
import sys
from src.logger import logging  # Custom logging module
from src.exception import CustomException  # Custom exception handler
import pandas as pd  # Library for data manipulation

# Importing the DataIngestion class from the relevant component
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Main execution block
if __name__ == '__main__':
    # Instantiate the DataIngestion object
    obj = DataIngestion()
    
    # Start the data ingestion process and store the paths of the train and test datasets
    train_data_path, test_data_path = obj.initiate_data_ingestion()  # Correct variable names
    
    # Output the paths of the train and test datasets
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()

    train_arr,test_arr,_ = data_transformation.initiate_data_transormation(train_data_path,test_data_path)


    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)