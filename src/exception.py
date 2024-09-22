import sys
from src.logger import logging  # Import the custom logging configuration from src.logger

# Function to get detailed error message
def error_message_detail(error, error_detail: sys):
    """
    Captures details about the error, including the file name and line number
    where the exception occurred.

    Parameters:
    - error: The exception that was raised.
    - error_detail: The sys module (provides access to some interpreter variables).

    Returns:
    - A formatted string with the script name, line number, and error message.
    """
    # Extract the traceback object (exception traceback details)
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename where the error occurred

    # Format the error message with filename, line number, and the error itself
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message  # Return the formatted error message

# Custom Exception class to handle and log custom exceptions
class CustomException(Exception):
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the custom exception with a detailed error message.

        Parameters:
        - error_message: The error message for the exception.
        - error_detail: Details about the exception (using sys to get traceback info).
        """
        super().__init__(error_message)  # Call the base Exception class initializer
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Get detailed error message

    def __str__(self):
        """
        Override the __str__ method to return the custom error message when
        the exception is printed or logged.
        """
        return self.error_message  # Return the detailed error message when the exception is raised

# Main block of code for testing logging and exception handling
# Uncomment this section for testing purposes

# if __name__ == "__main__":
#     logging.info("Logging has started")  # Log an informational message indicating that logging has started

#     try:
#         a = 1 / 0  # Example code that will raise a division by zero exception
#     except Exception as e:
#         logging.info('Division by zero')  # Log the occurrence of division by zero
#         raise CustomException(e, sys)  # Raise the custom exception with detailed error information
