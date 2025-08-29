# XposeAI - Fraud Detection Dashboard

A powerful, single-page FastAPI web dashboard for fraud detection across multiple transaction datasets. Built with modern web technologies for a smooth and responsive user experience.

## 🚀 Features

### 📊 Data Upload & Analysis
- **CSV Upload**: Drag & drop or click to upload transaction datasets
- **Auto-Detection**: Smart column mapping for credit card, mobile money, debit card transactions
- **Pattern Analysis**: Comprehensive fraud pattern detection with interactive visualizations
- **Real-time Preview**: Instant data preview and validation

### 🎯 Machine Learning
- **Pretrained Models**: Ready-to-use fraud detection models
- **Custom Training**: Train new models with your datasets
- **Multiple Algorithms**: Random Forest, XGBoost, Logistic Regression, SVM
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC evaluation
- **Model Persistence**: Save and load trained models automatically

### 🔬 Synthetic Data Generation
- **Realistic Patterns**: Generate synthetic fraud transaction data
- **Configurable Parameters**: Control fraud ratios, amount ranges, dataset types
- **Anomaly Features**: Include realistic fraud indicators and behavioral patterns
- **Multiple Formats**: Support for different transaction types

### 📈 Interactive Visualizations
- **Plotly Charts**: Beautiful, interactive fraud distribution and analysis charts
- **Real-time Updates**: Dynamic visualizations that update with your data
- **Responsive Design**: Charts adapt to different screen sizes

### 🛡️ Error Handling & Validation
- **File Validation**: Comprehensive CSV file validation (format, size, content)
- **User-Friendly Messages**: Clear error messages and guidance
- **Graceful Degradation**: Robust error handling for all operations

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **Python 3.11**: Modern Python with type hints
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Joblib**: Model serialization

### Frontend
- **Bootstrap 5**: Modern, responsive CSS framework
- **Vanilla JavaScript**: Clean, efficient client-side code
- **Plotly.js**: Interactive data visualizations
- **Bootstrap Icons**: Professional icon set

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- 1GB free disk space

### Supported Data Formats
- **File Format**: CSV only
- **File Size**: Maximum 50MB
- **Required Columns**: Amount (numeric), Fraud Label (binary)
- **Optional Columns**: Timestamp, Merchant, Location, Card Type, User ID

## 🚀 Quick Start

### One-Click Replit Deployment
1. Click the Run button in Replit
2. The FastAPI server will start automatically on port 5000
3. Access the dashboard at the provided URL

### Local Installation
```bash
# Clone the repository
git clone <repository-url>
cd xpose-ai

# Install dependencies (use pip or conda)
pip install fastapi uvicorn pandas scikit-learn plotly xgboost joblib matplotlib seaborn imbalanced-learn python-multipart jinja2 aiofiles

# Run the application
python main.py
```

The dashboard will be available at `http://localhost:5000`

## 📖 Usage Guide

### 1. Upload Data
1. Navigate to the dashboard
2. Drag & drop a CSV file or click "Choose File"
3. Review auto-detected column mappings or adjust manually
4. Click "Analyze Data" to process

### 2. View Analysis Results
- Interactive charts showing fraud distribution and patterns
- Key metrics: total transactions, fraud rate, average amounts
- Detailed list of detected fraud transactions

### 3. Train Models
- **Option A**: Load pretrained model for immediate use
- **Option B**: Train custom model with your data
  - Choose algorithm (Random Forest, XGBoost, Logistic Regression, SVM)
  - Adjust parameters (test size, random state, cross-validation)
  - Monitor training progress and results

### 4. Generate Synthetic Data
- Configure parameters: sample size, fraud ratio, dataset type
- Set amount ranges and enable advanced features
- Download generated data or use directly for training

### 5. Make Predictions
- Run predictions on current dataset with trained model
- View high-risk transactions (>70% fraud probability)
- Analyze model performance metrics

## 📊 Supported Dataset Types

### Credit Card Transactions
- Transaction amounts, merchant categories
- Card types, locations, timestamps
- Fraud indicators based on spending patterns

### Mobile Money Transactions
- Mobile payment providers, agent locations
- Transfer amounts, user demographics
- Fraud patterns specific to mobile payments

### Debit Card Transactions
- ATM withdrawals, POS transactions
- Account information, transaction locations
- Bank-specific fraud indicators

### Mixed Transaction Types
- Combined datasets with multiple payment methods
- Unified fraud detection across all transaction types

## 🔧 API Endpoints

The dashboard provides a REST API for programmatic access:

- `GET /` - Main dashboard page
- `POST /upload_csv` - Upload and validate CSV files
- `POST /analyze_data` - Analyze fraud patterns
- `POST /train_model` - Train custom fraud detection model
- `POST /load_pretrained` - Load pretrained model
- `POST /generate_synthetic` - Generate synthetic data
- `POST /predict` - Run predictions on dataset
- `GET /status` - Get application status

## 🛡️ Error Handling

The application includes comprehensive error handling:

- **File Upload Errors**: Invalid format, size limits, corrupted files
- **Data Processing Errors**: Missing columns, invalid data types, empty datasets
- **Model Training Errors**: Insufficient data, algorithm failures, memory issues
- **Prediction Errors**: Model compatibility, feature mismatches

All errors display user-friendly messages with guidance for resolution.

## 🎨 Design Features

- **Minimal, Neutral Design**: Clean interface focusing on functionality
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Bootstrap 5**: Modern, accessible UI components
- **Smooth Animations**: Subtle transitions and loading states
- **Consistent Color Scheme**: Professional blue-based palette
- **Intuitive Navigation**: Clear section organization with status indicators

## 🚨 Limitations

- **File Size**: Maximum 50MB per CSV file
- **Memory Usage**: Large datasets may require more RAM
- **Model Training**: Can be time-intensive for large datasets
- **Real-time Processing**: Not designed for streaming data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the error messages in the dashboard
2. Review the usage guide above
3. Create an issue in the repository

## 🔄 Version History

- **v1.0.0**: Initial release with full fraud detection pipeline
- FastAPI-based single-page dashboard
- Multi-algorithm model training
- Synthetic data generation
- Interactive visualizations
- Comprehensive error handling

---

**XposeAI** - Expose fraud with AI-powered detection and analysis.