import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import io
import base64
from flask import Flask, request, render_template, jsonify
import xgboost as xgb

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

# Initialize SHAP explainer
try:
    explainer = shap.TreeExplainer(model)
    shap_available = True
except Exception as e:
    print(f"Warning: SHAP initialization failed: {e}")
    print("SHAP interpretability will not be available")
    shap_available = False

def plot_to_base64(plt):
    """Convert matplotlib plot to base64 string for HTML embedding"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Convert to base64 to embed in HTML
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def get_feature_importance():
    """Generate feature importance plot for the model"""
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # For XGBoost models that don't have the attribute directly
        importance = model.get_booster().get_score(importance_type='weight')
        # Convert to array if it's a dictionary
        if isinstance(importance, dict):
            importance_array = np.zeros(len(feature_names))
            for key, value in importance.items():
                try:
                    idx = int(key.replace('f', ''))
                    if idx < len(importance_array):
                        importance_array[idx] = value
                except ValueError:
                    pass
            importance = importance_array

    # Create plot
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(sorted_idx)), importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return plot_to_base64(plt)

def get_shap_summary_plot(features_df):
    """Generate SHAP summary plot for the given features"""
    if not shap_available:
        return None
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(scaled_features)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, scaled_features, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    
    return plot_to_base64(plt)

def get_shap_decision_plot(features_df, idx=0):
    """Generate SHAP decision plot for a single instance"""
    if not shap_available:
        return None
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(scaled_features)
    
    # Create decision plot for the specified instance
    plt.figure(figsize=(10, 6))
    shap.decision_plot(explainer.expected_value, shap_values[idx], 
                      features=scaled_features[idx], feature_names=feature_names, show=False)
    plt.title('SHAP Decision Plot')
    plt.tight_layout()
    
    return plot_to_base64(plt)

def get_shap_force_plot(features_df, idx=0):
    """Generate SHAP force plot for a single instance"""
    if not shap_available:
        return None
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(scaled_features)
    
    # Create force plot for the specified instance
    plt.figure(figsize=(10, 3))
    shap.force_plot(explainer.expected_value, shap_values[idx], 
                   features=scaled_features[idx], feature_names=feature_names, 
                   matplotlib=True, show=False)
    plt.title('SHAP Force Plot')
    plt.tight_layout()
    
    return plot_to_base64(plt)

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html', feature_names=feature_names, 
                          shap_available=shap_available)

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
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale the features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Get interpretation if requested
        interpretation_method = request.form.get('interpretation_method', None)
        interpretations = {}
        
        if interpretation_method == 'feature_importance':
            interpretations['feature_importance'] = get_feature_importance()
        
        elif interpretation_method == 'shap' and shap_available:
            interpretations['shap_summary'] = get_shap_summary_plot(features_df)
            interpretations['shap_decision'] = get_shap_decision_plot(features_df)
            interpretations['shap_force'] = get_shap_force_plot(features_df)
        
        return jsonify({
            'prediction': float(prediction),
            'features': {name: float(value) for name, value in zip(feature_names, features)},
            'interpretations': interpretations
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
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
        
        # Get interpretation if requested
        interpretation_method = request.form.get('interpretation_method', None)
        interpretations = {}
        
        if interpretation_method == 'feature_importance':
            interpretations['feature_importance'] = get_feature_importance()
        
        elif interpretation_method == 'shap' and shap_available:
            interpretations['shap_summary'] = get_shap_summary_plot(features_df)
            # For CSV uploads, we can show the decision plot for the first row as an example
            if len(features_df) > 0:
                interpretations['shap_decision'] = get_shap_decision_plot(features_df, 0)
                interpretations['shap_force'] = get_shap_force_plot(features_df, 0)
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'interpretations': interpretations
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
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
    <title>Catboost Model Deployment with Interpretability</title>
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
        input, button, select {
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
        .interpretation-section {
            margin-top: 20px;
            overflow-x: auto;
        }
        .interpretation-section img {
            max-width: 100%;
            height: auto;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <h1>Catboost Model Deployment </h1>
    
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
                
                <div style="margin-top: 20px;">
                    <label for="interpretation-select">Model Interpretation Method:</label>
                    <select id="interpretation-select" name="interpretation_method">
                        <option value="none">None</option>
                        <option value="feature_importance">Feature Importance</option>
                        {% if shap_available %}
                        <option value="shap">SHAP Values</option>
                        {% endif %}
                    </select>
                </div>
                
                <button type="submit" style="margin-top: 20px;">Predict</button>
            </form>
            <div id="manual-result" class="result-section">
                <h3>Prediction Result</h3>
                <p id="prediction-value"></p>
                
                <div id="manual-interpretation" class="interpretation-section">
                    <div id="feature-importance-plot" style="display: none;">
                        <h4>Feature Importance</h4>
                        <img id="feature-importance-img" src="" alt="Feature Importance Plot">
                    </div>
                    
                    <div id="shap-plots" style="display: none;">
                        <h4>SHAP Summary Plot</h4>
                        <img id="shap-summary-img" src="" alt="SHAP Summary Plot">
                        
                        <h4>SHAP Decision Plot</h4>
                        <img id="shap-decision-img" src="" alt="SHAP Decision Plot">
                        
                        <h4>SHAP Force Plot</h4>
                        <img id="shap-force-img" src="" alt="SHAP Force Plot">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="form-section">
            <h2>CSV File Upload</h2>
            <p>Upload a CSV file with columns for each feature. The file should include headers matching the feature names.</p>
            <form id="csv-form" enctype="multipart/form-data">
                <input type="file" id="csv-file" name="file" accept=".csv" required>
                
                <div style="margin-top: 10px;">
                    <label for="csv-interpretation-select">Model Interpretation Method:</label>
                    <select id="csv-interpretation-select" name="interpretation_method">
                        <option value="none">None</option>
                        <option value="feature_importance">Feature Importance</option>
                        {% if shap_available %}
                        <option value="shap">SHAP Values</option>
                        {% endif %}
                    </select>
                </div>
                
                <button type="submit" style="margin-top: 10px;">Upload and Predict</button>
            </form>
            <div id="csv-result" class="result-section">
                <h3>Batch Prediction Results</h3>
                <p id="batch-summary"></p>
                <div id="batch-details"></div>
                
                <div id="csv-interpretation" class="interpretation-section">
                    <div id="csv-feature-importance-plot" style="display: none;">
                        <h4>Feature Importance</h4>
                        <img id="csv-feature-importance-img" src="" alt="Feature Importance Plot">
                    </div>
                    
                    <div id="csv-shap-plots" style="display: none;">
                        <h4>SHAP Summary Plot (All Rows)</h4>
                        <img id="csv-shap-summary-img" src="" alt="SHAP Summary Plot">
                        
                        <h4>SHAP Decision Plot (First Row Example)</h4>
                        <img id="csv-shap-decision-img" src="" alt="SHAP Decision Plot">
                        
                        <h4>SHAP Force Plot (First Row Example)</h4>
                        <img id="csv-shap-force-img" src="" alt="SHAP Force Plot">
                    </div>
                </div>
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
                    // Hide interpretation sections
                    document.getElementById('feature-importance-plot').style.display = 'none';
                    document.getElementById('shap-plots').style.display = 'none';
                } else {
                    document.getElementById('prediction-value').textContent = `Predicted value: ${data.prediction}`;
                    
                    // Handle interpretations
                    if (data.interpretations) {
                        // Feature importance
                        if (data.interpretations.feature_importance) {
                            document.getElementById('feature-importance-plot').style.display = 'block';
                            document.getElementById('feature-importance-img').src = data.interpretations.feature_importance;
                        } else {
                            document.getElementById('feature-importance-plot').style.display = 'none';
                        }
                        
                        // SHAP plots
                        if (data.interpretations.shap_summary) {
                            document.getElementById('shap-plots').style.display = 'block';
                            document.getElementById('shap-summary-img').src = data.interpretations.shap_summary;
                            document.getElementById('shap-decision-img').src = data.interpretations.shap_decision;
                            document.getElementById('shap-force-img').src = data.interpretations.shap_force;
                        } else {
                            document.getElementById('shap-plots').style.display = 'none';
                        }
                    } else {
                        // Hide all interpretation sections if no interpretations
                        document.getElementById('feature-importance-plot').style.display = 'none';
                        document.getElementById('shap-plots').style.display = 'none';
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('manual-result').style.display = 'block';
                document.getElementById('prediction-value').innerHTML = `<span class="error">Error: ${error.message}</span>`;
                // Hide interpretation sections
                document.getElementById('feature-importance-plot').style.display = 'none';
                document.getElementById('shap-plots').style.display = 'none';
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
                    // Hide interpretation sections
                    document.getElementById('csv-feature-importance-plot').style.display = 'none';
                    document.getElementById('csv-shap-plots').style.display = 'none';
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
                    
                    // Handle interpretations
                    if (data.interpretations) {
                        // Feature importance
                        if (data.interpretations.feature_importance) {
                            document.getElementById('csv-feature-importance-plot').style.display = 'block';
                            document.getElementById('csv-feature-importance-img').src = data.interpretations.feature_importance;
                        } else {
                            document.getElementById('csv-feature-importance-plot').style.display = 'none';
                        }
                        
                        // SHAP plots
                        if (data.interpretations.shap_summary) {
                            document.getElementById('csv-shap-plots').style.display = 'block';
                            document.getElementById('csv-shap-summary-img').src = data.interpretations.shap_summary;
                            document.getElementById('csv-shap-decision-img').src = data.interpretations.shap_decision;
                            document.getElementById('csv-shap-force-img').src = data.interpretations.shap_force;
                        } else {
                            document.getElementById('csv-shap-plots').style.display = 'none';
                        }
                    } else {
                        // Hide all interpretation sections if no interpretations
                        document.getElementById('csv-feature-importance-plot').style.display = 'none';
                        document.getElementById('csv-shap-plots').style.display = 'none';
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('csv-result').style.display = 'block';
                document.getElementById('batch-summary').innerHTML = `<span class="error">Error: ${error.message}</span>`;
                document.getElementById('batch-details').innerHTML = '';
                // Hide interpretation sections
                document.getElementById('csv-feature-importance-plot').style.display = 'none';
                document.getElementById('csv-shap-plots').style.display = 'none';
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
    print("Access the web interface at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)