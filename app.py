# Initialize Flask app
app = Flask(__name__)

# Global variables for the dataset and model
data = None
model = None
scaler = None
encoder = None
X_train = None
categorical_cols = []
numerical_cols = []
target_column = None

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Manufacturing Predictive Analysis API",
        "endpoints": {
            "/upload": "Upload a dataset (POST)",
            "/train": "Train the model (POST)",
            "/predict": "Make predictions (POST)"
        }
    }), 200

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_data():
    global data, categorical_cols, numerical_cols, encoder, target_column

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    # Load CSV file into a DataFrame
    data = pd.read_csv(file)

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # Identify the target column
    for column in data.columns:
        if "flag" in column or "Flag" in column or "target" in column.lower():
            target_column = column
            break

    if target_column is None:
        return jsonify({"error": "Target column not found in the dataset."}), 400

    # Handle categorical columns with One-Hot Encoding
    if categorical_cols:
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)  # Ensure target column is not encoded

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)

    # Update numerical columns after encoding
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    return jsonify({"message": "Dataset uploaded successfully", "columns": data.columns.tolist()}), 200

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global data, model, scaler, numerical_cols, target_column, X_train

    if data is None:
        return jsonify({"error": "No dataset uploaded. Use the /upload endpoint first."}), 400

    # Check if the target column exists in the DataFrame
    if target_column not in data.columns:
        return jsonify({"error": f"Target column '{target_column}' not found in the dataset."}), 400

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Dynamically update numerical_cols after dropping the target column
    numerical_cols = [col for col in X.select_dtypes(include=[np.number]).columns]

    # Scale numerical columns
    scaler = StandardScaler()

    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return jsonify({
        "message": "Model trained successfully",
        "accuracy": accuracy,
        "classification_report": report
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, encoder, X_train, numerical_cols, categorical_cols, target_column

    if model is None:
        return jsonify({"error": "Model is not trained. Use the /train endpoint first."}), 400

    try:
        # Use 'X_train' and drop the target column (as you need features for prediction)
        input_df = X_train

        # Get the first row of the input data used for prediction
        prediction_input = input_df.iloc[0].to_dict()

        # Make prediction
        prediction = model.predict(input_df)
        confidence = max(model.predict_proba(input_df)[0])

        # Check the prediction and map it to "Yes" or "No"
        prediction_result = "Yes" if prediction[0] == 1 else "No"

        return jsonify({
            "Prediction_Input": prediction_input,
            "Downtime": prediction_result,
            "Confidence": confidence
        }), 200

    except Exception as e:
        import traceback
        return jsonify({"error": traceback.format_exc()}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
