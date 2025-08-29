import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import io

def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded CSV file."""
    try:
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size (limit to 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            return False, "File size too large. Please upload a file smaller than 50MB."
        
        # Check file extension
        if not uploaded_file.name.lower().endswith('.csv'):
            return False, "Invalid file format. Please upload a CSV file."
        
        # Try to read the file
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, nrows=10)  # Read first 10 rows for validation
            uploaded_file.seek(0)  # Reset for actual use
        except Exception as e:
            return False, f"Failed to read CSV file: {str(e)}"
        
        # Check if file has data
        if df.empty:
            return False, "Uploaded file is empty"
        
        # Check minimum number of columns
        if len(df.columns) < 2:
            return False, "File must have at least 2 columns"
        
        return True, "File validation successful"
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def get_sample_data_info() -> str:
    """Return information about expected data formats."""
    return """
    **Expected Data Formats:**
    
    **Required Columns:**
    - Amount: Transaction amount (numeric)
    - Fraud Label: Binary indicator (0/1, True/False, etc.)
    - Timestamp: Date/time of transaction (optional)
    
    **Optional Columns:**
    - Merchant: Merchant name/category
    - Location: Transaction location
    - Card Type: Payment method type
    - User ID: Customer/account identifier
    
    **Supported Dataset Types:**
    - Credit Card transactions
    - Mobile Money transfers
    - Debit Card transactions
    - Mixed transaction types
    
    **File Requirements:**
    - CSV format only
    - Maximum size: 50MB
    - Minimum 2 columns
    - UTF-8 encoding recommended
    """

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    try:
        return f"${amount:,.2f}"
    except:
        return str(amount)

def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    try:
        return f"{value:.2%}"
    except:
        return str(value)

def calculate_basic_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate basic statistics for a column."""
    try:
        if column not in df.columns:
            return {}
        
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if len(series) == 0:
            return {}
        
        return {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75)
        }
        
    except Exception as e:
        return {'error': str(e)}

def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect common data quality issues."""
    try:
        issues = {
            'missing_values': {},
            'duplicate_rows': 0,
            'negative_amounts': 0,
            'zero_amounts': 0,
            'outliers': {},
            'data_types': {}
        }
        
        # Missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'][col] = missing_count
        
        # Duplicate rows
        issues['duplicate_rows'] = df.duplicated().sum()
        
        # Amount-related issues
        amount_columns = [col for col in df.columns if 'amount' in col.lower()]
        for col in amount_columns:
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                issues['negative_amounts'] += (numeric_col < 0).sum()
                issues['zero_amounts'] += (numeric_col == 0).sum()
                
                # Outliers using IQR method
                Q1 = numeric_col.quantile(0.25)
                Q3 = numeric_col.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((numeric_col < (Q1 - 1.5 * IQR)) | 
                               (numeric_col > (Q3 + 1.5 * IQR))).sum()
                issues['outliers'][col] = outlier_count
            except:
                continue
        
        # Data types
        for col in df.columns:
            issues['data_types'][col] = str(df[col].dtype)
        
        return issues
        
    except Exception as e:
        return {'error': str(e)}

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for DataFrame."""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        return csv_string
    except Exception as e:
        return f"Error creating download: {str(e)}"

def validate_model_input(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """Validate DataFrame for model input."""
    try:
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for sufficient data
        if len(df) < 10:
            return False, "Insufficient data. Need at least 10 rows for model training."
        
        # Check for fraud labels
        if 'is_fraud' in df.columns:
            fraud_labels = df['is_fraud'].unique()
            if len(fraud_labels) < 2:
                return False, "Dataset must contain both fraud and legitimate transactions."
        
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def generate_summary_report(df: pd.DataFrame, analysis_results: Dict) -> str:
    """Generate a text summary report of the analysis."""
    try:
        report = []
        report.append("=== FRAUD DETECTION ANALYSIS REPORT ===\n")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"Total Transactions: {len(df):,}")
        report.append(f"Number of Columns: {len(df.columns)}")
        report.append(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}" 
                     if 'timestamp' in df.columns else "Date Range: Not available")
        report.append("")
        
        # Fraud statistics
        if 'fraud_count' in analysis_results:
            fraud_count = analysis_results['fraud_count']
            total_count = len(df)
            fraud_rate = (fraud_count / total_count) * 100
            
            report.append("FRAUD STATISTICS:")
            report.append(f"Fraud Transactions: {fraud_count:,}")
            report.append(f"Legitimate Transactions: {total_count - fraud_count:,}")
            report.append(f"Fraud Rate: {fraud_rate:.2f}%")
            report.append("")
        
        # Amount analysis
        if 'fraud_amount_stats' in analysis_results:
            fraud_stats = analysis_results['fraud_amount_stats']
            report.append("AMOUNT ANALYSIS:")
            report.append(f"Average Fraud Amount: ${fraud_stats['mean']:.2f}")
            report.append(f"Total Fraud Amount: ${fraud_stats['total']:,.2f}")
            report.append(f"Median Fraud Amount: ${fraud_stats['median']:.2f}")
            report.append("")
        
        # Top fraud patterns
        if 'top_fraud_merchant' in analysis_results:
            report.append("TOP FRAUD MERCHANTS:")
            for merchant, count in list(analysis_results['top_fraud_merchant'].items())[:5]:
                report.append(f"- {merchant}: {count} transactions")
            report.append("")
        
        report.append("=== END OF REPORT ===")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating report: {str(e)}"
