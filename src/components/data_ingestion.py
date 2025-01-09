# Import required libraries and modules
import os
import sys
from src.logger import logging  # Custom logging module for logging messages
from src.exception import CustomException  # Custom exception handler
import pandas as pd  # Library for data manipulation
from sklearn.model_selection import train_test_split  # Function to split data into training and test sets
from dataclasses import dataclass  # For creating configuration classes

# Define the configuration for data ingestion using a dataclass
@dataclass
class DataIngestionconfig:
    """
    DataIngestionconfig stores the file paths where raw, train, and test data will be saved.
    The paths are defined under the 'artifacts' directory to store processed data.
    """
    train_data_path = os.path.join('artifacts', 'train.csv')  # Path for training data
    test_data_path = os.path.join('artifacts', 'test.csv')  # Path for test data
    raw_data_path = os.path.join('artifacts', 'raw.csv')  # Path for raw data

# Create a class for the Data Ingestion process
class DataIngestion:
    """
    DataIngestion class handles the process of reading raw data, splitting it into training and test sets,
    and saving the processed data for further use in model training and testing.
    """
    def __init__(self):
        # Initialize the DataIngestionconfig to access data paths
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process which involves:
        1. Reading the raw dataset from the specified location.
        2. Saving the raw data into a designated path.
        3. Splitting the dataset into training and test sets (75% training, 25% test).
        4. Saving the split data to respective files.
        
        Returns:
            Tuple: Paths to the training data and test data CSV files.
        """
        logging.info('Data Ingestion method starts')  # Log the start of the data ingestion process

        try:
            # Read the dataset from the specified path (source dataset for gem pricing)
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')  # Log that the dataset was successfully read

            # Create the directory to store raw data, if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved at the specified path')  # Log raw data storage

            logging.info("Train Test Split initiated")  # Log the train-test split process

            # Split the dataset into training and test sets (75% train, 25% test)
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=69)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info('Training data saved successfully')  # Log training data storage

            # Save the test set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Test data saved successfully')  # Log test data storage

            logging.info('Ingestion of Data is Completed')  # Log the completion of data ingestion

            # Return the paths of the train and test data for further use in the pipeline
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Log the error and raise a custom exception with the error details
            logging.info('Error occurred in Data Ingestion process')  
            raise CustomException(e, sys)
