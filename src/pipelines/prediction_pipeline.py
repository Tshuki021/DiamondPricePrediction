import sys
import os
from src.exception import CustomException  # Custom exception handler for better error handling
from src.logger import logging  # Custom logger for tracking process logs
from src.utils import load_object  # Utility function to load saved objects
import pandas as pd  # Library for data manipulation

# Class for handling the prediction pipeline
class PredictPipeline:
    """
    The PredictPipeline class is responsible for loading the pre-trained model 
    and preprocessor to make predictions based on input features.
    """
    
    def __init__(self):
        pass

    def predict(self, features):
        """
        This method loads the preprocessor and model, applies preprocessing to the input features, 
        and returns the predicted values.
        
        Args:
        features (pd.DataFrame): A pandas DataFrame containing input features.
        
        Returns:
        np.ndarray: An array of predicted values.
        """
        try:
            # Load preprocessor and model from their respective file paths
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict the output using the trained model
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Exception occurred in prediction process')
            raise CustomException(e, sys)


# Class to handle custom input data for the prediction
class CustomData:
    """
    The CustomData class serves as a structure to gather custom input data, 
    convert it to a pandas DataFrame, and provide it to the prediction pipeline.
    """
    
    def __init__(self, carat: float, depth: float, table: float, x: float, y: float, z: float, 
                 cut: str, color: str, clarity: str):
        """
        Initializes the CustomData object with diamond features.
        
        Args:
        carat (float): Carat weight of the diamond.
        depth (float): Depth percentage of the diamond.
        table (float): Table percentage of the diamond.
        x (float): Length of the diamond in mm.
        y (float): Width of the diamond in mm.
        z (float): Height of the diamond in mm.
        cut (str): Cut quality of the diamond.
        color (str): Color grade of the diamond.
        clarity (str): Clarity grade of the diamond.
        """
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        """
        Converts the custom data attributes into a pandas DataFrame for model input.
        
        Returns:
        pd.DataFrame: A DataFrame containing the input features.
        """
        try:
            # Create a dictionary of input features
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            # Convert dictionary to a pandas DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered for Prediction')
            return df
        except Exception as e:
            logging.info('Exception occurred while gathering data for prediction pipeline')
            raise CustomException(e, sys)
