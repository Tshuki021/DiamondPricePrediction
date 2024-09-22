import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Handling Feature Scaling & Ordinal Encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # For combining two pipelines
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## Data Transformation config

@dataclass
class DataTransformationconfig:
    """
    Data class to store configuration related to data transformation.
    Stores the file path where the preprocessor object will be saved as a pickle file.
    """
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class for performing data transformation operations, including:
    1. Defining pipelines for numerical and categorical feature processing.
    2. Applying transformations to training and test data.
    3. Saving the transformation object as a pickle file.
    """
    def __init__(self):
        # Initialize the configuration for data transformation
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        """
        Creates and returns a ColumnTransformer object that applies the following:
        - Numerical Pipeline: Imputes missing values with the median and scales the features.
        - Categorical Pipeline: Imputes missing values with the most frequent category, ordinal encodes the features, and scales them.
        
        Returns:
            preprocessor (ColumnTransformer): A preprocessor object combining both numerical and categorical transformations.
        """
        try:
            logging.info('Data Transformation Initiated')

            # Define numerical and categorical columns
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns = ['cut', 'color', 'clarity']

            # Define custom ranking for ordinal encoding
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline: Handles missing values and scales numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
                    ('scaler', StandardScaler())  # Scale the features to mean 0 and standard deviation 1
                ]
            )

            # Categorical Pipeline: Handles missing values, encodes categorical features, and scales them
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),  # Ordinal encoding based on custom categories
                    ('scaler', StandardScaler())  # Scale the encoded categorical features
                ]
            )

            # Combine the pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),  # Apply numerical transformations
                ('cat_pipeline', cat_pipeline, categorical_columns)  # Apply categorical transformations
            ])

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Reads the train and test datasets, applies the transformation, and saves the preprocessor object as a pickle file.
        
        Args:
            train_data_path (str): Path to the training data CSV.
            test_data_path (str): Path to the testing data CSV.
        
        Returns:
            Tuple: Transformed train array, test array, and path to the saved preprocessor object.
        """
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Reading train and test data completed')
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # Defining target column and columns to drop
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            # Separating input and target features for both train and test datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying the transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')

            # Combine the transformed input features with the target feature for both train and test data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object as a pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file created')

            # Return the transformed arrays and the preprocessor file path
            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation')
            raise CustomException(e, sys)
