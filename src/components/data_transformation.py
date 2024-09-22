import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder ## Handling Feature Scaling & Ordinal Encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer ## For combining two pipelines
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')
    

## Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns =  ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns =  ['cut', 'color', 'clarity']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pinpeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Step 1: Fill missing values with the median of the column
                    ('scaler', StandardScaler())  # Step 2: Scale the numerical features to have a mean of 0 and a standard deviation of 1
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Step 1: Fill missing values with the most frequent category in the column
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),  # Step 2: Encode categorical variables into numerical values based on predefined categories
                    ('scaler', StandardScaler())  # Step 3: Scale the encoded categorical features (may or may not be necessary depending on the model)
                ]
            )

            # Preprocessor: Combines the numerical and categorical pipelines into one
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),  # Apply the numerical pipeline to the specified numerical columns
                ('cat_pipeline', cat_pipeline, categorical_columns)  # Apply the categorical pipeline to the specified categorical columns
            ])

            return preprocessor
        
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)


    def initiate_data_transormation(self,train_data_path,test_date_path):
        try:
            # reading train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_date_path)

            logging.info('Reading train anf test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## apply the transformation

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            logging.info('Preprocessor pickle file is created')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            logging.info('Exception occured in the initiate_datatransformation')
            raise CustomException(e,sys)