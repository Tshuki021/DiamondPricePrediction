# Import necessary modules
from flask import Flask, request, render_template  # Flask for web framework, request for handling HTTP methods, render_template for rendering HTML templates
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline  # Importing custom classes for data handling and prediction

# Initialize Flask application
application = Flask(__name__)  # Create a Flask instance
app = application  # Alias for the app

@app.route('/')
def home_page():
    """
    Renders the homepage of the application.
    
    Returns:
    HTML template for the index page.
    """
    return render_template('index.html')  # Renders the home page (index.html)

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles the prediction request. 
    If it's a GET request, it shows the form page for user input. 
    If it's a POST request, it processes the form data and returns the prediction result.
    
    Returns:
    - If GET: Renders the form page to collect data.
    - If POST: Renders the form page again but with prediction results.
    """
    if request.method == 'GET':  # If it's a GET request
        return render_template('form.html')  # Render the form to collect input data
    
    else:  # If it's a POST request
        # Collect data from the form fields
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        
        # Convert form data into a DataFrame for the prediction pipeline
        final_new_data = data.get_data_as_dataframe()
        
        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Make a prediction using the pipeline
        pred = predict_pipeline.predict(final_new_data)

        # Round the predicted result to 2 decimal places
        results = round(pred[0], 2)

        # Render the form page again, this time passing the prediction result
        return render_template('form.html', final_result=results)

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)  # Start the Flask app with debugging enabled. Use Ctrl+C to stop the app.

# To stop the Flask application, use 'ctrl+c' in the terminal.