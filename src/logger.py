import logging
import os
from datetime import datetime

# Create a log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path for storing logs (current working directory + /logs/ + log file name)
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory for logs if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Log output will be saved to this file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO  # Set logging level to INFO for generic information logging
)
