# XposeAI - Fraud Detection Dashboard

XposeAI is a comprehensive Streamlit-based fraud detection dashboard that provides powerful tools for analyzing transaction data, detecting fraud patterns, and training machine learning models.

## Features

### 🔍 Data Analysis
- **CSV Upload**: Upload transaction datasets with automatic column detection and mapping
- **Multi-format Support**: Supports credit card, mobile money, debit card, and mixed transaction types
- **Smart Column Mapping**: Automatically detects common column patterns or allows manual mapping
- **Pattern Analysis**: Comprehensive fraud pattern analysis with detailed statistics

### 📊 Visualization
- **Interactive Charts**: Real-time visualizations using Plotly for fraud patterns
- **Distribution Analysis**: Amount distributions, fraud timelines, and correlation heatmaps
- **Merchant Analysis**: Fraud breakdown by merchant categories and locations
- **Time-based Analysis**: Fraud patterns by hour, day, and seasonal trends

### 🎯 Machine Learning
- **Pretrained Models**: Ready-to-use fraud detection models
- **Custom Training**: Train new models with your own datasets
- **Multiple Algorithms**: Support for Random Forest, XGBoost, Logistic Regression, and SVM
- **Performance Metrics**: Comprehensive evaluation with Accuracy, Precision, Recall, F1-Score, and AUC
- **Model Persistence**: Save and load trained models using joblib

### 🔬 Synthetic Data Generation
- **Realistic Data**: Generate synthetic fraud transaction data for testing
- **Configurable Parameters**: Control fraud ratios, amount ranges, and dataset types
- **Anomaly Features**: Include realistic fraud indicators and patterns
- **Multiple Formats**: Support for different transaction types and patterns

## Installation

### Requirements
- Python 3.7+
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- NumPy
- XGBoost
- Imbalanced-learn
- Joblib

### Setup
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn plotly numpy xgboost imbalanced-learn joblib
   ```
3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Usage

### 1. Data Upload & Analysis
1. Navigate to the "📊 Data Upload & Analysis" section
2. Upload your CSV file containing transaction data
3. Review the auto-detected column mappings or adjust manually
4. View the analysis results including fraud statistics and patterns

### 2. Model Training
1. Go to "🎯 Model Training" section
2. Choose between using the pretrained model or training a new one
3. Configure training parameters (algorithm, test size, cross-validation)
4. Review training results and performance metrics

### 3. Synthetic Data Generation
1. Access "🔬 Synthetic Data Generator" section
2. Configure generation parameters (samples, fraud ratio, dataset type)
3. Generate synthetic data with realistic fraud patterns
4. Download or use the generated data for model training

### 4. Results & Metrics
1. Visit "📈 Results & Metrics" section
2. Review detailed analysis results and visualizations
3. View model performance metrics
4. Run predictions on current dataset
5. Download results and predictions

## Supported Data Formats

### Required Columns
- **Amount**: Transaction amount (numeric)
- **Fraud Label**: Binary fraud indicator (0/1, True/False, fraud/legitimate)

### Optional Columns
- **Timestamp**: Transaction date/time
- **Merchant**: Merchant name or category
- **Location**: Transaction location
- **Card Type**: Payment method type
- **User ID**: Customer or account identifier

### Dataset Types
- Credit Card transactions
- Mobile Money transfers
- Debit Card transactions
- Mixed transaction types

## Model Algorithms

The dashboard supports multiple machine learning algorithms:

1. **Random Forest**: Robust ensemble method, good for mixed data types
2. **XGBoost**: Gradient boosting, excellent performance on structured data
3. **Logistic Regression**: Fast and interpretable linear model
4. **Support Vector Machine (SVM)**: Effective for high-dimensional data

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Error Handling

The application includes comprehensive error handling for:
- Invalid file uploads
- Missing or incorrect column formats
- Data processing errors
- Model training failures
- Visualization errors

All errors display user-friendly messages with guidance for resolution.

## File Structure

