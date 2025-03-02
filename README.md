# Multivariate Regression API with Model Interpretation

## Overview

This project provides a comprehensive pipeline for supervised multivariate regression using **XGBoost**. The system facilitates exploratory data analysis (EDA), robust preprocessing, model training, hyperparameter tuning, and interpretability through **SHAP** and **Feature Importance** techniques. The API serves predictions via a Flask-based interface, allowing single-instance queries and batch processing through CSV uploads.

### Purpose of Each Section:
- **EDA**: Extract statistical insights, detect correlations, analyze distributions, and identify outliers.
- **Preprocessing**: Ensure data quality, feature selection, scaling, and partitioning.
- **Model Training & Selection**: Optimize and validate models through hyperparameter tuning and cross-validation.
- **Evaluation**: Assess performance using various regression metrics, emphasizing error minimization and robustness.
- **Deployment & API**: Enable model inference through REST API endpoints.
- **Interpretability**: Utilize **SHAP** and **Feature Importance** to enhance explainability in decision-making.

## Setup and Execution

### Clone Repository

```sh
git clone <repository-url>
cd <repository-folder>
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run API Server

```sh
python api.py
```

### Access API

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your local browser.

## API Endpoints

### Predict Single Instance

**POST** `/predict`

#### Request Example:

```json
{
  "feature_0": 1.5,
  "feature_1": 3.2,
  "interpretation_method": "shap"
}
```

### Predict from CSV

**POST** `/predict_csv`

#### Upload CSV file containing required features.

### Interpretation Methods

- **Feature Importance**: Ranks input features by predictive significance.
- **SHAP Values**: Allocates contributions of each feature to an individual prediction.
- **None**: Outputs only numerical predictions.

## Exploratory Data Analysis (EDA)

- **Summary Statistics**: Compute count, mean, standard deviation, quartiles, and missing values.
- **Correlation Matrix**: Analyze feature dependencies using Pearson, Spearman, and Kendall coefficients.
- **Variance & Covariance**: Identify variability across features and their linear relationships.
- **Distribution Analysis**: Visualize data behavior using histograms, KDE, and boxplots.
- **Principal Component Analysis (PCA)**: Reduce dimensionality while preserving variance.
- **Clustering Techniques**: Evaluate K-Means and DBSCAN for feature grouping.
- **Outlier Detection**: Apply **Z-score** and **IQR-based** methods.

![Correlation Techniques](data/eda/correlations/correlation_methods_comparison.png)
![Variance](data\eda\variance\variance_barplot.png")
![PCA Analysis]("data\eda\PCA\pca_loadings_heatmap.png")
![Feature Distributions](data/notebook_plot_8.png)


## Preprocessing (ETL Pipeline)

- **Data Cleaning**: Handle missing values and outliers.
- **Feature Selection**: Define predictors (`X`) and regression target (`Y`).
- **Train-Test Splitting**: Divide dataset for generalization testing.
- **Standardization**: Scale features to normalize distributions.

## Model Training & Selection

- **Optuna Optimization**:
  - Bayesian search for hyperparameter tuning.
  - Objective: **Minimize MSE** using **100 trials** with **5-fold cross-validation**.
- **BayesSearchCV**:
  - Grid search over Bayesian-optimized hyperparameters.
  - K-fold cross-validation ensures robustness.

#### Sample Visualizations
![Hyperparameter Optimization](data/notebook_plot_22.png)
![Model Convergence](data/notebook_plot_25.png)

## Model Evaluation

Regression performance metrics:

- **Mean Squared Error (MSE)**: Quadratic penalty for large errors.
- **Root Mean Squared Error (RMSE)**: Measures error in the original scale.
- **Mean Absolute Error (MAE)**: Average absolute deviations.
- **Explained Variance Score**: Percentage of variance explained by the model.
- **Mean Absolute Percentage Error (MAPE)**: Relative percentage error against actual values.

#### Sample Visualizations
![Residual Analysis](data/notebook_plot_30.png)
![Prediction vs Actual](data/notebook_plot_35.png)

## Model Deployment

- Flask-based REST API for local inference.
- **Interactive UI**: Enables direct input and visualization.
- **Batch Prediction**: Supports CSV-based bulk processing.

## Model Interpretation

### Feature Importance Plot

- Evaluates feature weight impact on predictions.
- Extracted directly from the XGBoost model.

### SHAP Summary & Decision Plots

- **SHAP Summary Plot**: Global feature attribution.
- **SHAP Decision Plot**: Visualizes feature impact on specific predictions.
- **SHAP Force Plot**: Displays instance-level influence.

#### Sample Visualizations
![Feature Importance](data/notebook_plot_40.png)
![SHAP Summary Plot](data/notebook_plot_45.png)
![SHAP Decision Plot](data/notebook_plot_50.png)

## Solutions Architecture Evolution

This section outlines the iterative development of the solution, detailing how architectural decisions were refined to enhance model accuracy, interpretability, and efficiency.

### Initial Approach
- Basic regression model with default parameters.
- Minimal feature engineering and preprocessing.

### Intermediate Enhancements
- Introduction of **Optuna** for hyperparameter tuning.
- Enhanced preprocessing with outlier handling and feature selection.
- Cross-validation added for robustness.

### Finalized Architecture
- **XGBoost** selected for performance and interpretability.
- **SHAP-based explanations** integrated for model insights.
- **Flask API** built for real-time predictions and batch inference.

## References

- Peña, "Análisis de Datos Multivariantes"
- Suresh, "Hands-On Exploratory Data Analysis with Python"
- James, "An Introduction to Statistical Learning"
- Hastie, "The Elements of Statistical Learning"
- Schölkopf, "Learning with Kernels"
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"

