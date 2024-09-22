---

# Diamond Price Prediction

## Introduction

This project aims to predict the price of a diamond using various independent variables that describe the diamond's physical attributes. The data used for this project is sourced from [Kaggle's Playground Series S3E8](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv).

### Dataset Features

**Independent Variables:**
1. `id`: Unique identifier of each diamond.
2. `carat`: Weight of the diamond (in carats).
3. `cut`: Quality of the diamond cut.
4. `color`: Color grade of the diamond.
5. `clarity`: Measure of the purity of the diamond.
6. `depth`: Height of the diamond from culet to table.
7. `table`: Width of the top facet of the diamond.
8. `x`: Length of the diamond in mm.
9. `y`: Width of the diamond in mm.
10. `z`: Depth of the diamond in mm.

**Target Variable:**
- `price`: The price of the given diamond.

### Data Insights
- No outliers, missing values, or duplicates were found in the dataset.
- Numerical and categorical columns were segregated.
- The dimensions (`x`, `y`, `z`) are closely correlated with `carat`.
- Domain information for attributes such as `cut`, `clarity`, and `color` was utilized from the [American Gem Society](https://www.americangemsociety.org/buying-diamonds-with-confidence/ags-diamond-grading-system/).

---

## Project Structure

```
├── artifacts
│   ├── model.pkl               # Serialized machine learning model.
│   ├── preprocessor.pkl         # Preprocessing pipeline for transforming data.
│   ├── raw.csv                  # Raw dataset.
│   ├── test.csv                 # Test dataset.
│   ├── train.csv                # Training dataset.
│
├── notebooks
│   ├── EDA.ipynb                # Exploratory Data Analysis notebook.
│   ├── Model_Training.ipynb      # Model training and evaluation notebook.
│
├── src
│   ├── exception.py             # Custom exception handling module.
│   ├── logger.py                # Logger configuration for tracking project logs.
│   ├── utils.py                 # Utility functions for model evaluation, saving, and loading objects.
│
├── templates
│   ├── form.html                # HTML form for user input to predict diamond prices.
│   ├── index.html               # Homepage of the web application.
│
├── .gitignore                   # Files and directories to be ignored in version control.
├── README.md                    # Project documentation.
├── application.py               # Flask web application for predicting diamond prices.
├── requirements.txt             # Required Python packages for the project.
├── setup.py                     # Setup configuration for the project.
```

---

## Key Components

### 1. Notebooks
- **EDA.ipynb**: 
  - Conducts exploratory data analysis.
  - Identifies key relationships between variables (e.g., dimensions `x`, `y`, and `z` are highly correlated with `carat`).
  - Handles the mapping of categorical features such as `cut`, `clarity`, and `color`.

- **Model_Training.ipynb**:
  - Preprocesses data using pipelines:
    - Numerical columns: `SimpleImputer` (median imputation).
    - Categorical columns: `SimpleImputer` (most frequent) + `OrdinalEncoder`.
    - Both pipelines use `StandardScaler` for scaling.
  - Trains multiple regression models including:
    - Linear Regression
    - Lasso
    - Ridge
    - ElasticNet
    - Decision Tree Regressor
  - Evaluates model performance using `r2_score`.

### 2. `src` Directory
- **exception.py**: Contains custom exception classes and functions for error handling throughout the project.
- **logger.py**: Configures logging for capturing essential runtime information.
- **utils.py**: Includes utility functions to evaluate models, save trained models, and load them for predictions.

### 3. Web Application (`application.py`)
- Built using Flask.
- Provides a user interface to input diamond attributes and predict the diamond price using the trained model.

### 4. Templates
- **index.html**: Homepage of the web application.
- **form.html**: Form to input diamond attributes for prediction.

### 5. Requirements
- **requirements.txt**: Lists all the dependencies and libraries required for the project, including Flask, scikit-learn, pandas, etc.

---

## Model Performance

During model training, multiple regression models were evaluated based on the `r2_score` metric. Below are the results for each model:

```plaintext
Model Report : {'LinearRegression': 0.9344523096065713, 'Lasso': 0.934468560543525, 'Ridge': 0.9344522047348359, 'ElasticNet': 0.8536173350575074, 'DecisionTree': 0.9581454114033142}
Best Model Model name : DecisionTree, R2 Score : 0.9581454114033142
```

The **Decision Tree Regressor** was selected as the best-performing model with an R² score of **0.9581**.

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diamond-price-prediction.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd diamond-price-prediction
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python application.py
   ```

---

## Usage

1. Access the web application through your browser at `http://127.0.0.1:5000`.
2. Input diamond attributes through the form provided.
3. Get the predicted price based on the trained machine learning model.

---

## Results
- The **Decision Tree Regressor** was found to be the best model for predicting the price of diamonds with an R² score of **0.9581**.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
