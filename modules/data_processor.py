import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, Tuple, Optional, List

class DataProcessor:
    """Handles data processing, validation, and column mapping for fraud detection."""
    
    def __init__(self):
        self.common_column_patterns = {
            'amount': ['amount', 'value', 'transaction_amount', 'trans_amount', 'sum', 'price'],
            'fraud_label': ['fraud', 'is_fraud', 'fraudulent', 'label', 'class', 'target', 'isfraud'],
            'timestamp': ['timestamp', 'date', 'time', 'transaction_date', 'trans_date', 'created_at'],
            'merchant': ['merchant', 'vendor', 'shop', 'store', 'business', 'merchant_name'],
            'location': ['location', 'city', 'state', 'country', 'region', 'zip', 'zipcode'],
            'card_type': ['card_type', 'card', 'payment_method', 'type', 'category'],
            'user_id': ['user_id', 'customer_id', 'account', 'id', 'userid', 'customerid']
        }
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect column mappings based on column names and data patterns."""
        detected = {}
        columns_lower = [col.lower() for col in df.columns]
        
        for target_col, patterns in self.common_column_patterns.items():
            best_match = None
            best_score = 0
            
            for i, col_name in enumerate(columns_lower):
                for pattern in patterns:
                    # Exact match gets highest score
                    if col_name == pattern:
                        best_match = df.columns[i]
                        best_score = 100
                        break
                    # Partial match
                    elif pattern in col_name or col_name in pattern:
                        score = len(pattern) / len(col_name) * 50
                        if score > best_score:
                            best_match = df.columns[i]
                            best_score = score
                
                if best_score == 100:
                    break
            
            if best_match and best_score > 30:
                # Additional validation based on data type and content
                if self._validate_column_content(df[best_match], target_col):
                    detected[target_col] = best_match
        
        return detected
    
    def _validate_column_content(self, series: pd.Series, expected_type: str) -> bool:
        """Validate if column content matches expected type."""
        try:
            if expected_type == 'amount':
                # Should be numeric and positive
                numeric_series = pd.to_numeric(series, errors='coerce')
                return not numeric_series.isna().all() and (numeric_series >= 0).any()
            
            elif expected_type == 'fraud_label':
                # Should be binary (0/1 or True/False or similar)
                unique_vals = set(series.dropna().astype(str).str.lower().unique())
                binary_patterns = [
                    {'0', '1'}, {'true', 'false'}, {'yes', 'no'}, 
                    {'fraud', 'legitimate'}, {'fraud', 'normal'}
                ]
                return any(unique_vals.issubset(pattern) or pattern.issubset(unique_vals) 
                          for pattern in binary_patterns)
            
            elif expected_type == 'timestamp':
                # Try to parse as datetime
                try:
                    pd.to_datetime(series.dropna().iloc[:100])
                    return True
                except:
                    return False
            
            elif expected_type in ['merchant', 'location', 'card_type']:
                # Should be categorical/string data
                return series.dtype == 'object' and len(series.unique()) > 1
            
            elif expected_type == 'user_id':
                # Should have reasonable number of unique values
                unique_ratio = len(series.unique()) / len(series)
                return unique_ratio > 0.1  # At least 10% unique values
            
            return True
            
        except Exception:
            return False
    
    def process_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Optional[pd.DataFrame]:
        """Process raw data according to column mapping."""
        try:
            processed_df = df.copy()
            
            # Rename columns according to mapping
            rename_dict = {v: k for k, v in column_mapping.items() if v and v in df.columns}
            processed_df = processed_df.rename(columns=rename_dict)
            
            # Process amount column
            if 'amount' in processed_df.columns:
                processed_df['amount'] = pd.to_numeric(processed_df['amount'], errors='coerce')
                processed_df = processed_df.dropna(subset=['amount'])
                processed_df = processed_df[processed_df['amount'] >= 0]
            else:
                raise ValueError("Amount column is required but not found")
            
            # Process fraud label
            if 'fraud_label' in processed_df.columns:
                processed_df['is_fraud'] = self._standardize_fraud_label(processed_df['fraud_label'])
                processed_df = processed_df.dropna(subset=['is_fraud'])
            else:
                raise ValueError("Fraud label column is required but not found")
            
            # Process timestamp if available
            if 'timestamp' in processed_df.columns:
                try:
                    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
                    processed_df['hour'] = processed_df['timestamp'].dt.hour
                    processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
                    processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6])
                except:
                    processed_df = processed_df.drop(columns=['timestamp'])
            
            # Process categorical columns
            categorical_cols = ['merchant', 'location', 'card_type']
            for col in categorical_cols:
                if col in processed_df.columns:
                    # Encode categorical variables
                    processed_df[f'{col}_encoded'] = pd.Categorical(processed_df[col]).codes
                    processed_df[f'{col}_frequency'] = processed_df[col].map(processed_df[col].value_counts())
            
            # Create additional features
            processed_df = self._create_features(processed_df)
            
            # Remove original label column and keep standardized version
            if 'fraud_label' in processed_df.columns:
                processed_df = processed_df.drop(columns=['fraud_label'])
            
            return processed_df
            
        except Exception as e:
            raise Exception(f"Data processing failed: {str(e)}")
    
    def _standardize_fraud_label(self, series: pd.Series) -> pd.Series:
        """Standardize fraud labels to binary 0/1 format."""
        series_str = series.astype(str).str.lower().str.strip()
        
        # Map various representations to binary
        fraud_mapping = {
            'true': 1, 'false': 0,
            '1': 1, '0': 0,
            'yes': 1, 'no': 0,
            'fraud': 1, 'legitimate': 0, 'normal': 0,
            'fraudulent': 1, 'genuine': 0
        }
        
        result = series_str.map(fraud_mapping)
        
        # If mapping didn't work, try numeric conversion
        if result.isna().all():
            result = pd.to_numeric(series, errors='coerce')
            result = (result > 0).astype(int)
        
        return result
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better fraud detection."""
        try:
            # Amount-based features
            if 'amount' in df.columns:
                df['amount_log'] = np.log1p(df['amount'])
                df['amount_rounded'] = (df['amount'] % 1 == 0).astype(int)
                
                # Amount percentiles
                df['amount_percentile'] = df['amount'].rank(pct=True)
                df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
            
            # Time-based features (if timestamp available)
            if 'hour' in df.columns:
                df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
                df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Frequency-based features
            for col in ['merchant_frequency', 'location_frequency']:
                if col in df.columns:
                    df[f'{col}_log'] = np.log1p(df[col])
                    df[f'is_rare_{col.split("_")[0]}'] = (df[col] <= 5).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Warning: Feature creation failed: {str(e)}")
            return df
    
    def analyze_fraud_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze fraud patterns in the processed data."""
        try:
            if 'is_fraud' not in df.columns:
                raise ValueError("Fraud label column 'is_fraud' not found")
            
            fraud_transactions = df[df['is_fraud'] == 1].copy()
            legitimate_transactions = df[df['is_fraud'] == 0].copy()
            
            analysis = {
                'fraud_transactions': fraud_transactions,
                'legitimate_transactions': legitimate_transactions,
                'fraud_count': len(fraud_transactions),
                'legitimate_count': len(legitimate_transactions),
                'fraud_rate': len(fraud_transactions) / len(df) if len(df) > 0 else 0
            }
            
            # Amount analysis
            if 'amount' in df.columns:
                analysis['fraud_amount_stats'] = {
                    'mean': fraud_transactions['amount'].mean() if len(fraud_transactions) > 0 else 0,
                    'median': fraud_transactions['amount'].median() if len(fraud_transactions) > 0 else 0,
                    'std': fraud_transactions['amount'].std() if len(fraud_transactions) > 0 else 0,
                    'total': fraud_transactions['amount'].sum() if len(fraud_transactions) > 0 else 0
                }
                
                analysis['legitimate_amount_stats'] = {
                    'mean': legitimate_transactions['amount'].mean() if len(legitimate_transactions) > 0 else 0,
                    'median': legitimate_transactions['amount'].median() if len(legitimate_transactions) > 0 else 0,
                    'std': legitimate_transactions['amount'].std() if len(legitimate_transactions) > 0 else 0,
                    'total': legitimate_transactions['amount'].sum() if len(legitimate_transactions) > 0 else 0
                }
            
            # Time analysis
            if 'hour' in df.columns:
                fraud_by_hour = fraud_transactions['hour'].value_counts().sort_index()
                analysis['fraud_by_hour'] = fraud_by_hour.to_dict()
            
            # Categorical analysis
            categorical_cols = ['merchant', 'location', 'card_type']
            for col in categorical_cols:
                if col in df.columns:
                    fraud_by_category = fraud_transactions[col].value_counts().head(10)
                    analysis[f'top_fraud_{col}'] = fraud_by_category.to_dict()
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Pattern analysis failed: {str(e)}")
