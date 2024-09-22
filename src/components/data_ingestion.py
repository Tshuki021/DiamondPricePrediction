# Import required libraries and modules
import os
import sys
from src.logger import logging  # Custom logging module for logging messages
from src.exception import CustomException  # Custom exception handler
import pandas as pd  # Library for data manipulation
from sklearn.model_selection import train_test_split  # Function to split data into training and test sets
from dataclasses import dataclass  # For creating configuration classes

# Define the configuration for data ingestion using a dataclass
# This will store paths for raw, train, and test data
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')  # Path for training data
    test_data_path = os.path.join('artifacts', 'test.csv')  # Path for test data
    raw_data_path = os.path.join('artifacts', 'raw.csv')  # Path for raw data

# Create a class for the Data Ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the DataIngestionconfig to access data paths
        self.ingestion_config = DataIngestionconfig()

    # Method to start the data ingestion process
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')  # Log the start of the process

        try:
            # Read the dataset from the specified path
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')  # Log that the dataset was successfully read

            # Create the directory to store raw data, if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train Test Split")  # Log the train-test split process

            # Split the dataset into training and test sets (75% train, 25% test)
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=69)

            # Save the training and test sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is Completed')  # Log the completion of data ingestion

            # Return the paths of the train and test data for further use
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Handle any exceptions that occur during the data ingestion process
        except Exception as e:
            logging.info('Error occurred in Data Ingestion config')  # Log the error
            raise CustomException(e, sys)  # Raise a custom exception with error details
