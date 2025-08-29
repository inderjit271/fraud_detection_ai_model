# XposeAI - Fraud Detection Dashboard

## Overview

XposeAI is a comprehensive Streamlit-based fraud detection dashboard that provides end-to-end capabilities for analyzing transaction data, detecting fraud patterns, and training machine learning models. The application supports multiple transaction types (credit card, mobile money, debit card), offers automated data processing with intelligent column mapping, and includes synthetic data generation for testing and training purposes.

The system is designed as a modular web application that enables users to upload transaction datasets, perform exploratory data analysis, train custom fraud detection models, and visualize results through interactive charts and metrics dashboards.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page navigation
- **Layout**: Wide layout with expandable sidebar for section navigation
- **State Management**: Session state for maintaining data, analysis results, and trained models across user interactions
- **UI Components**: Interactive forms, file upload widgets, data tables, and chart displays

### Backend Architecture
- **Modular Design**: Separated into distinct modules for data processing, model training, synthetic data generation, visualization, and utilities
- **Data Processing Pipeline**: Automated column detection and mapping system with support for multiple transaction formats
- **Model Training System**: Multi-algorithm support (Random Forest, XGBoost, Logistic Regression, SVM) with cross-validation and performance metrics
- **Synthetic Data Generation**: Configurable fraud pattern simulation with realistic transaction characteristics

### Data Processing
- **Column Mapping**: Intelligent pattern-based detection for common transaction fields (amount, fraud_label, timestamp, merchant, location, etc.)
- **Data Validation**: File size limits (50MB), format validation, and content verification
- **Feature Engineering**: Automatic encoding of categorical variables, scaling of numerical features, and handling of missing values
- **Preprocessing Pipeline**: StandardScaler for numerical features and LabelEncoder for categorical variables

### Machine Learning Pipeline
- **Model Support**: Multiple algorithms with hyperparameter optimization
- **Training Strategy**: Train-test split with stratified sampling for imbalanced datasets
- **Evaluation Metrics**: Comprehensive performance assessment including accuracy, precision, recall, F1-score, and AUC
- **Model Persistence**: Joblib-based model serialization for saving and loading trained models
- **Cross-validation**: StratifiedKFold validation for robust performance estimation

### Visualization System
- **Interactive Charts**: Plotly-based visualizations for fraud distribution, amount analysis, and temporal patterns
- **Chart Types**: Pie charts, histograms, box plots, correlation heatmaps, and time series plots
- **Color Scheme**: Consistent color palette for fraud (red) vs legitimate (green) transactions
- **Responsive Design**: Charts adapt to different screen sizes and data volumes

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas**: Data manipulation and analysis library for transaction data processing
- **NumPy**: Numerical computing for mathematical operations and array handling
- **Scikit-learn**: Machine learning algorithms, preprocessing, and evaluation metrics

### Machine Learning
- **XGBoost**: Gradient boosting framework for advanced fraud detection models
- **Imbalanced-learn**: Specialized tools for handling imbalanced fraud datasets
- **Joblib**: Model serialization and parallel processing utilities

### Visualization
- **Plotly**: Interactive plotting library for fraud analysis charts and dashboards
- **Seaborn**: Statistical data visualization (used as fallback for certain chart types)
- **Matplotlib**: Base plotting library for chart generation

### Data Processing
- **Datetime**: Built-in Python module for timestamp handling and temporal analysis
- **Regular Expressions (re)**: Pattern matching for intelligent column detection
- **IO**: File handling utilities for CSV upload and processing

### Development Tools
- **Warnings**: Python warnings management to suppress non-critical alerts
- **Typing**: Type hints for better code documentation and IDE support
- **Random**: Random number generation for synthetic data creation

The application is designed to work with CSV files containing transaction data and does not require external database connections or API integrations. All processing is performed locally within the Streamlit environment.