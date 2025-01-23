# Manufacturing Predictive Analysis

## Overview
This project provides a RESTful API that predicts machine downtime or production defects based on manufacturing data. The API allows users to upload data, train a logistic regression model, and make predictions for machine downtime (Yes/No). The model is trained on a dataset containing features like machine temperature, run time, pressure, vibration levels, and maintenance schedule. This solution leverages Python with Flask, scikit-learn for machine learning, and integrates an API for easy access to the predictive model.

## Architecture
The architecture of the solution is simple and follows the typical model-train-predict workflow:

1. **Data Generation**: A synthetic manufacturing dataset is created with key features like Machine_ID, Temperature, Run_Time, Pressure, Vibration_Level, and Maintenance_Schedule.
2. **Preprocessing**: The dataset is processed using One-Hot Encoding for categorical features and scaling for numerical features. 
3. **Model Training**: Logistic Regression is used to train the model for predicting machine downtime based on the features.
4. **API Endpoints**: 
   - `/upload`: Uploads the dataset.
   - `/train`: Trains the machine learning model.
   - `/predict`: Provides predictions based on new input.

## Dataset
A synthetic dataset is generated to simulate manufacturing operations. This dataset includes:

- **Machine_ID**: Unique identifier for each machine (1-100).
- **Temperature**: Random temperature values between 50 and 100 degrees.
- **Run_Time**: Random runtime values between 10 and 200 minutes.
- **Pressure**: Random pressure values between 20 and 150.
- **Vibration_Level**: Random vibration levels between 0 and 10.
- **Maintenance_Schedule**: Random maintenance schedule (`Yes`/`No`).
- **Downtime_Flag**: Binary target flag (1 for downtime, 0 for no downtime).

The data is saved in a CSV file: `manufacturing_data.csv`.

## Logistic Regression Model
- **Logistic Regression** is used as the classifier. It is a simple yet effective supervised machine learning algorithm, ideal for binary classification tasks like predicting machine downtime.
- The model is trained using features such as Temperature, Run_Time, Pressure, and Vibration_Level.
- **Model Evaluation**: Accuracy, F1-score, and classification report are used to evaluate the performance of the model.

## API Functions

### 1. `/upload` (POST)
**Description**: Uploads a CSV file containing manufacturing data. The file is processed to handle categorical columns (One-Hot Encoding) and scale numerical features.
- **Request**: `POST /upload`
- **Body**: `multipart/form-data` with the key `file` (CSV file).
- **Response**:
  - Success: `{"message": "Dataset uploaded successfully", "columns": [...]}`.
  - Error: `{"error": "No file provided"}` if no file is provided.

### 2. `/train` (POST)
**Description**: Trains the logistic regression model using the uploaded dataset. It returns performance metrics like accuracy and classification report.
- **Request**: `POST /train`
- **Response**:
  - Success: `{"message": "Model trained successfully", "accuracy": ..., "classification_report": {...}}`.
  - Error: `{"error": "No dataset uploaded. Use the /upload endpoint first."}` if no dataset has been uploaded.

### 3. `/predict` (POST)
**Description**: Accepts input data for prediction and returns the prediction result along with the confidence score.
- **Request**: `POST /predict`
- **Body**: `{"Temperature": ..., "Run_Time": ..., "Pressure": ..., "Vibration_Level": ...}`
- **Response**:
  - Success: `{"Prediction_Input": {...}, "Downtime": "Yes", "Confidence": 0.85}`.
  - Error: `{"error": "Model is not trained. Use the /train endpoint first."}` if the model is not trained yet.

## How to Set Up and Use the API

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/manufacturing-predictive-analysis.git
cd manufacturing-predictive-analysis
```

### 2. Install Dependencies
Make sure you have Python installed. Then, install the necessary dependencies:
```bash
pip install -r requirements.txt
```
This will install:
- `Flask`: For creating the RESTful API.
- `scikit-learn`: For machine learning and model training.
- `pandas`: For data handling.
- `numpy`: For data manipulation.

### 3. Run the API
To start the Flask app, run the following command:
```bash
python app.py
```
This will start the API on `http://127.0.0.1:5000`.

### 4. Testing the Endpoints
You can test the API using **Postman** or **cURL**.

#### Upload Dataset:
- **Method**: POST
- **URL**: `http://127.0.0.1:5000/upload`
- **Body**: Choose `form-data` and select a file to upload.

#### Train the Model:
- **Method**: POST
- **URL**: `http://127.0.0.1:5000/train`
- After uploading the dataset, train the model by sending a POST request.

#### Make Predictions:
- **Method**: POST
- **URL**: `http://127.0.0.1:5000/predict`
- **Body**: Send a JSON object with the input features:
```json
{
  "Temperature": 80,
  "Run_Time": 120,
  "Pressure": 100,
  "Vibration_Level": 5
}
```

### 5. Example cURL Commands
- **Upload Data**:
```bash
curl -X POST -F "file=@manufacturing_data.csv" http://127.0.0.1:5000/upload
```

- **Train Model**:
```bash
curl -X POST http://127.0.0.1:5000/train
```

- **Predict Downtime**:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120, "Pressure": 100, "Vibration_Level": 5}' http://127.0.0.1:5000/predict
```

### 6. Storing the Dataset
You can upload any dataset in CSV format with columns such as `Machine_ID`, `Temperature`, `Run_Time`, `Pressure`, `Vibration_Level`, and `Downtime_Flag`. Make sure that the dataset has a target column related to downtime, such as `Downtime_Flag`.

### 7. Model Evaluation
Once the model is trained, the API will return a classification report with metrics like precision, recall, F1-score, and confusion matrix to assess its performance.

## Additional Notes
- **Scaling**: Numerical features are scaled using `StandardScaler` to ensure better model performance.
- **One-Hot Encoding**: Categorical features are encoded using `OneHotEncoder` to convert them into numerical format.
- **Logistic Regression**: A simple logistic regression model is used for prediction due to its simplicity and effectiveness for binary classification tasks.

## Conclusion
This API provides a simple and extensible solution for predictive analysis of manufacturing operations. It allows users to upload their dataset, train a machine learning model, and make predictions about machine downtime, helping businesses optimize their operations and reduce unplanned downtime.
