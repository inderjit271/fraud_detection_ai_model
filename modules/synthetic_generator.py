import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class SyntheticGenerator:
    """Generates synthetic fraud transaction data for training and testing."""
    
    def __init__(self):
        self.merchants = {
            'credit_card': ['Amazon', 'Walmart', 'Target', 'Best Buy', 'Home Depot', 'Gas Station', 'Restaurant', 'Hotel'],
            'mobile_money': ['MTN Mobile', 'Airtel Money', 'Orange Money', 'M-Pesa', 'Tigo Cash', 'Mobile Transfer', 'Agent'],
            'debit_card': ['ATM Withdrawal', 'POS Terminal', 'Online Store', 'Grocery Store', 'Pharmacy', 'Bank Transfer']
        }
        
        self.locations = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle'
        ]
        
        self.card_types = ['Visa', 'Mastercard', 'American Express', 'Discover']
        
        # Fraud patterns based on research
        self.fraud_patterns = {
            'high_amount': {'min_multiplier': 3, 'max_multiplier': 10},
            'unusual_time': {'night_hours': [0, 1, 2, 3, 4, 5], 'weekend_preference': 0.7},
            'merchant_preference': {'rare_merchants': 0.6, 'high_risk_merchants': ['Gas Station', 'ATM Withdrawal']},
            'location_anomaly': {'foreign_location_prob': 0.3},
            'frequency_anomaly': {'burst_transactions': 0.4}
        }
    
    def generate_fraud_data(self, num_samples: int = 1000, fraud_ratio: float = 0.1,
                           dataset_type: str = 'credit_card', amount_range: Tuple[int, int] = (10, 5000),
                           include_anomalies: bool = True, seasonal_patterns: bool = True,
                           correlation_features: bool = True) -> pd.DataFrame:
        """Generate synthetic fraud transaction dataset."""
        try:
            num_fraud = int(num_samples * fraud_ratio)
            num_legitimate = num_samples - num_fraud
            
            # Generate base data
            data = []
            
            # Generate legitimate transactions
            for i in range(num_legitimate):
                transaction = self._generate_legitimate_transaction(
                    dataset_type, amount_range, seasonal_patterns
                )
                transaction['is_fraud'] = 0
                transaction['transaction_id'] = f"TXN_{i:06d}"
                data.append(transaction)
            
            # Generate fraudulent transactions
            for i in range(num_fraud):
                transaction = self._generate_fraud_transaction(
                    dataset_type, amount_range, include_anomalies, seasonal_patterns
                )
                transaction['is_fraud'] = 1
                transaction['transaction_id'] = f"FRAUD_{i:06d}"
                data.append(transaction)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Add correlated features if requested
            if correlation_features:
                df = self._add_correlated_features(df)
            
            # Add noise and realistic variations
            df = self._add_realistic_variations(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Synthetic data generation failed: {str(e)}")
    
    def _generate_legitimate_transaction(self, dataset_type: str, amount_range: Tuple[int, int],
                                       seasonal_patterns: bool) -> Dict:
        """Generate a legitimate transaction."""
        transaction = {}
        
        # Amount follows normal distribution for legitimate transactions
        mean_amount = (amount_range[0] + amount_range[1]) / 2
        std_amount = (amount_range[1] - amount_range[0]) / 6
        transaction['amount'] = max(amount_range[0], 
                                  min(amount_range[1], 
                                      np.random.normal(mean_amount, std_amount)))
        
        # Timestamp (business hours preference)
        transaction['timestamp'] = self._generate_timestamp(seasonal_patterns, is_fraud=False)
        
        # Merchant based on dataset type
        merchants = self.merchants.get(dataset_type, self.merchants['credit_card'])
        transaction['merchant_category'] = np.random.choice(merchants)
        
        # Location
        transaction['location'] = np.random.choice(self.locations)
        
        # Card type (for card-based transactions)
        if dataset_type in ['credit_card', 'debit_card']:
            transaction['card_type'] = np.random.choice(self.card_types)
        
        # User ID
        transaction['user_id'] = f"USER_{np.random.randint(1000, 9999)}"
        
        return transaction
    
    def _generate_fraud_transaction(self, dataset_type: str, amount_range: Tuple[int, int],
                                  include_anomalies: bool, seasonal_patterns: bool) -> Dict:
        """Generate a fraudulent transaction with suspicious patterns."""
        transaction = {}
        
        # Fraud transactions often have unusual amounts
        if include_anomalies and random.random() < 0.4:
            # High amount fraud
            multiplier = np.random.uniform(
                self.fraud_patterns['high_amount']['min_multiplier'],
                self.fraud_patterns['high_amount']['max_multiplier']
            )
            transaction['amount'] = min(amount_range[1], amount_range[1] * multiplier / 3)
        else:
            # Round amount (common in fraud)
            if random.random() < 0.6:
                transaction['amount'] = float(np.random.choice([100, 200, 500, 1000, 1500, 2000]))
            else:
                mean_amount = (amount_range[0] + amount_range[1]) / 2
                transaction['amount'] = np.random.uniform(amount_range[0], amount_range[1])
        
        # Unusual timing for fraud
        transaction['timestamp'] = self._generate_timestamp(seasonal_patterns, is_fraud=True)
        
        # Merchant preferences for fraud
        merchants = self.merchants.get(dataset_type, self.merchants['credit_card'])
        if include_anomalies and random.random() < self.fraud_patterns['merchant_preference']['rare_merchants']:
            # Prefer high-risk merchants
            high_risk = [m for m in merchants if m in self.fraud_patterns['merchant_preference']['high_risk_merchants']]
            if high_risk:
                transaction['merchant_category'] = np.random.choice(high_risk)
            else:
                transaction['merchant_category'] = np.random.choice(merchants)
        else:
            transaction['merchant_category'] = np.random.choice(merchants)
        
        # Location anomalies
        if include_anomalies and random.random() < self.fraud_patterns['location_anomaly']['foreign_location_prob']:
            foreign_locations = ['Dubai', 'London', 'Tokyo', 'Mumbai', 'Lagos', 'Moscow', 'Unknown']
            transaction['location'] = np.random.choice(foreign_locations)
        else:
            transaction['location'] = np.random.choice(self.locations)
        
        # Card type
        if dataset_type in ['credit_card', 'debit_card']:
            transaction['card_type'] = np.random.choice(self.card_types)
        
        # User ID (fraud often involves compromised accounts)
        transaction['user_id'] = f"USER_{np.random.randint(1000, 9999)}"
        
        return transaction
    
    def _generate_timestamp(self, seasonal_patterns: bool, is_fraud: bool = False) -> datetime:
        """Generate realistic timestamps with seasonal and fraud patterns."""
        # Base date range (last 6 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Random date
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        if is_fraud:
            # Fraud transactions more likely at unusual times
            if random.random() < 0.4:
                # Night hours
                hour = np.random.choice(self.fraud_patterns['unusual_time']['night_hours'])
            else:
                hour = random.randint(0, 23)
            
            # Weekend preference for fraud
            if random.random() < self.fraud_patterns['unusual_time']['weekend_preference']:
                # Adjust to weekend
                days_to_weekend = (5 - random_date.weekday()) % 7
                if days_to_weekend < 2:
                    random_date += timedelta(days=days_to_weekend)
        else:
            # Legitimate transactions prefer business hours
            if random.random() < 0.7:
                hour = random.randint(9, 17)  # Business hours
            else:
                hour = random.randint(6, 22)  # Extended hours
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return random_date.replace(hour=hour, minute=minute, second=second)
    
    def _add_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that correlate with fraud patterns."""
        try:
            # Velocity features (transaction frequency)
            df['transactions_last_hour'] = np.random.poisson(2, len(df))
            df.loc[df['is_fraud'] == 1, 'transactions_last_hour'] = np.random.poisson(8, sum(df['is_fraud']))
            
            # Distance from home location
            df['distance_from_home'] = np.random.exponential(50, len(df))
            df.loc[df['is_fraud'] == 1, 'distance_from_home'] = np.random.exponential(200, sum(df['is_fraud']))
            
            # Account age (newer accounts more vulnerable)
            df['account_age_days'] = np.random.exponential(365, len(df))
            df.loc[df['is_fraud'] == 1, 'account_age_days'] = np.random.exponential(180, sum(df['is_fraud']))
            
            # Previous failed attempts
            df['failed_attempts_last_day'] = np.random.poisson(0.5, len(df))
            df.loc[df['is_fraud'] == 1, 'failed_attempts_last_day'] = np.random.poisson(3, sum(df['is_fraud']))
            
            # Merchant risk score
            merchant_risk = {
                'Gas Station': 0.8, 'ATM Withdrawal': 0.7, 'Online Store': 0.6,
                'Restaurant': 0.3, 'Grocery Store': 0.2, 'Pharmacy': 0.2
            }
            df['merchant_risk_score'] = df['merchant_category'].map(
                lambda x: merchant_risk.get(x, 0.4)
            )
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not add correlated features: {str(e)}")
            return df
    
    def _add_realistic_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add noise and realistic variations to make data more authentic."""
        try:
            # Add small random variations to amounts
            noise = np.random.normal(0, 0.01, len(df))
            df['amount'] = df['amount'] * (1 + noise)
            df['amount'] = df['amount'].round(2)
            
            # Ensure amounts are positive
            df['amount'] = df['amount'].abs()
            
            # Add timestamp variations
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add some missing values to make it realistic
            missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
            optional_cols = ['card_type', 'location']
            for col in optional_cols:
                if col in df.columns:
                    col_missing = np.random.choice(missing_indices, size=len(missing_indices)//2, replace=False)
                    df.loc[col_missing, col] = np.nan
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not add realistic variations: {str(e)}")
            return df
