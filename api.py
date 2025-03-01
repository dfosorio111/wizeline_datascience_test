import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
import io

app = Flask(__name__)

# Load the pre-trained scaler and model
scaler_dir = "data/gold/scaler"
scaler_path = os.path.join(scaler_dir, "standard_scaler.pkl")
scaler = joblib.load(scaler_path)

# Load the trained model
model_path = "data/gold/models/optuna/best_xgb_model.pkl"
model = joblib.load(model_path)

# Define the feature names
feature_names = [
    'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4',
    'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
    'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
    'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19'
]

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request from manual input."""
    try:
        # Get values from form
        features = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for {feature}'})
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}. Must be a number.'})
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features': {name: float(value) for name, value in zip(feature_names, features)}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Handle prediction request from CSV file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Read the CSV file
        content = file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            return jsonify({'error': 'Invalid CSV format'})
        
        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features in CSV: {", ".join(missing_features)}'})
        
        # Extract features in the correct order
        features_df = df[feature_names]
        
        # Scale features
        scaled_features = scaler.transform(features_df)
        
        # Make predictions
        predictions = model.predict(scaled_features)
        
        # Prepare results
        results = []
        for i, pred in enumerate(predictions):
            row_data = {name: float(features_df.iloc[i][name]) for name in feature_names}
            row_data['prediction'] = float(pred)
            results.append(row_data)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy'})

# Create templates directory and HTML file
def create_templates():
    """Create required template files if they don't exist."""
    os.makedirs('templates', exist_ok=True)
    
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>XGBoost Model Deployment</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .form-section {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
        }
        .feature-inputs {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }
        .feature-input {
            display: flex;
            flex-direction: column;
        }
        input, button {
            padding: 8px;
            margin-top: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            padding: 10px 15px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result-section {
            margin-top: 20px;
            display: none;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
    <h1>XGBoost Model Deployment</h1>
    
    <div class="container">
        <div class="form-section">
            <h2>Manual Feature Input</h2>
            <form id="manual-form">
                <div class="feature-inputs">
                    {% for feature in feature_names %}
                    <div class="feature-input">
                        <label for="{{ feature }}">{{ feature }}</label>
                        <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" required>
                    </div>
                    {% endfor %}
                </div>
                <button type="submit" style="margin-top: 20px;">Predict</button>
            </form>
            <div id="manual-result" class="result-section">
                <h3>Prediction Result</h3>
                <p id="prediction-value"></p>
            </div>
        </div>
        
        <div class="form-section">
            <h2>CSV File Upload</h2>
            <p>Upload a CSV file with columns for each feature. The file should include headers matching the feature names.</p>
            <form id="csv-form" enctype="multipart/form-data">
                <input type="file" id="csv-file" name="file" accept=".csv" required>
                <button type="submit" style="margin-top: 10px;">Upload and Predict</button>
            </form>
            <div id="csv-result" class="result-section">
                <h3>Batch Prediction Results</h3>
                <p id="batch-summary"></p>
                <div id="batch-details"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('manual-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('manual-result');
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    document.getElementById('prediction-value').innerHTML = `<span class="error">Error: ${data.error}</span>`;
                } else {
                    document.getElementById('prediction-value').textContent = `Predicted value: ${data.prediction}`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('manual-result').style.display = 'block';
                document.getElementById('prediction-value').innerHTML = `<span class="error">Error: ${error.message}</span>`;
            });
        });
        
        document.getElementById('csv-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/predict_csv', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('csv-result');
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    document.getElementById('batch-summary').innerHTML = `<span class="error">Error: ${data.error}</span>`;
                    document.getElementById('batch-details').innerHTML = '';
                } else {
                    document.getElementById('batch-summary').textContent = `Processed ${data.count} rows successfully.`;
                    
                    // Display first 5 predictions
                    const displayCount = Math.min(5, data.predictions.length);
                    let detailsHtml = `<h4>Sample of results (showing ${displayCount} of ${data.count}):</h4>`;
                    
                    if (displayCount > 0) {
                        detailsHtml += '<table border="1" style="border-collapse: collapse; width: 100%;">';
                        
                        // Create header row
                        detailsHtml += '<tr><th>Row</th><th>Prediction</th></tr>';
                        
                        // Add data rows
                        for (let i = 0; i < displayCount; i++) {
                            detailsHtml += `<tr>
                                <td>${i+1}</td>
                                <td>${data.predictions[i].prediction}</td>
                            </tr>`;
                        }
                        
                        detailsHtml += '</table>';
                    }
                    
                    document.getElementById('batch-details').innerHTML = detailsHtml;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('csv-result').style.display = 'block';
                document.getElementById('batch-summary').innerHTML = `<span class="error">Error: ${error.message}</span>`;
                document.getElementById('batch-details').innerHTML = '';
            });
        });
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    create_templates()
    print("Starting the model deployment server...")
    print("Access the web interface at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)