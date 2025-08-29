import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from modules.data_processor import DataProcessor
from modules.model_trainer import ModelTrainer
from modules.synthetic_generator import SyntheticGenerator
from modules.visualizer import Visualizer
from modules.utils import validate_file, get_sample_data_info

# Configure page
st.set_page_config(
    page_title="XposeAI - Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

def main():
    # Header
    st.title("🔍 XposeAI - Fraud Detection Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Select Section:",
            ["📊 Data Upload & Analysis", "🎯 Model Training", "🔬 Synthetic Data Generator", "📈 Results & Metrics"]
        )
    
    # Main content area
    if section == "📊 Data Upload & Analysis":
        data_upload_section()
    elif section == "🎯 Model Training":
        model_training_section()
    elif section == "🔬 Synthetic Data Generator":
        synthetic_data_section()
    elif section == "📈 Results & Metrics":
        results_section()

def data_upload_section():
    st.header("📊 Data Upload & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Transaction Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your transaction dataset. Supported formats: credit card, mobile money, debit card transactions."
        )
        
        if uploaded_file is not None:
            try:
                # Validate file
                is_valid, message = validate_file(uploaded_file)
                if not is_valid:
                    st.error(message)
                    return
                
                # Load data
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
                
                # Display sample data
                st.subheader("Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Column mapping
                st.subheader("Column Mapping")
                processor = DataProcessor()
                detected_mapping = processor.auto_detect_columns(data)
                
                if detected_mapping:
                    st.success("✅ Columns automatically detected!")
                    for key, value in detected_mapping.items():
                        st.write(f"**{key.title()}**: {value}")
                    
                    if st.button("Proceed with Auto-Detection", type="primary"):
                        st.session_state.column_mapping = detected_mapping
                        st.session_state.data = data
                        st.success("Column mapping saved!")
                else:
                    st.warning("⚠️ Could not auto-detect columns. Please map manually:")
                    manual_mapping = {}
                    
                    required_columns = ['amount', 'fraud_label', 'timestamp']
                    optional_columns = ['merchant', 'location', 'card_type', 'user_id']
                    
                    for col in required_columns:
                        manual_mapping[col] = st.selectbox(
                            f"Select column for {col.title()} (Required):",
                            [''] + list(data.columns),
                            key=f"map_{col}"
                        )
                    
                    for col in optional_columns:
                        manual_mapping[col] = st.selectbox(
                            f"Select column for {col.title()} (Optional):",
                            [''] + list(data.columns),
                            key=f"map_{col}"
                        )
                    
                    if st.button("Apply Manual Mapping", type="primary"):
                        # Validate required mappings
                        missing_required = [col for col in required_columns if not manual_mapping.get(col)]
                        if missing_required:
                            st.error(f"❌ Missing required columns: {', '.join(missing_required)}")
                        else:
                            st.session_state.column_mapping = {k: v for k, v in manual_mapping.items() if v}
                            st.session_state.data = data
                            st.success("Manual mapping applied!")
                
                # Analysis section
                if st.session_state.data is not None and st.session_state.column_mapping:
                    st.subheader("Quick Analysis")
                    
                    processor = DataProcessor()
                    processed_data = processor.process_data(st.session_state.data, st.session_state.column_mapping)
                    
                    if processed_data is not None:
                        analysis_results = processor.analyze_fraud_patterns(processed_data)
                        st.session_state.analysis_results = analysis_results
                        
                        # Display key metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("Total Transactions", len(processed_data))
                        with col_b:
                            fraud_count = len(analysis_results['fraud_transactions'])
                            st.metric("Fraud Transactions", fraud_count)
                        with col_c:
                            fraud_rate = (fraud_count / len(processed_data)) * 100
                            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                        with col_d:
                            avg_fraud_amount = analysis_results['fraud_transactions']['amount'].mean() if fraud_count > 0 else 0
                            st.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")
                        
                        # Show fraud transactions
                        if fraud_count > 0:
                            st.subheader("Detected Fraud Transactions")
                            st.dataframe(
                                analysis_results['fraud_transactions'].head(20),
                                use_container_width=True
                            )
                        else:
                            st.info("No fraud transactions detected in the dataset.")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        st.subheader("Dataset Information")
        st.info(get_sample_data_info())

def model_training_section():
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Options")
        
        model_option = st.radio(
            "Choose model option:",
            ["Use Pretrained Model", "Train New Model"]
        )
        
        if model_option == "Use Pretrained Model":
            st.info("Using pretrained fraud detection model trained on multiple transaction types.")
            
            if st.button("Load Pretrained Model", type="primary"):
                try:
                    # Load pretrained model
                    model = joblib.load('models/pretrained_model.joblib')
                    st.session_state.trained_model = model
                    st.success("✅ Pretrained model loaded successfully!")
                except:
                    st.error("❌ Pretrained model not found. Training a basic model...")
                    # Create a basic model if pretrained doesn't exist
                    trainer = ModelTrainer()
                    model = trainer.create_basic_model()
                    st.session_state.trained_model = model
                    st.success("✅ Basic model created!")
        
        else:  # Train New Model
            if st.session_state.data is None:
                st.warning("⚠️ Please upload and process a dataset first.")
                return
            
            st.subheader("Training Configuration")
            
            # Model parameters
            col_a, col_b = st.columns(2)
            with col_a:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("Random State", value=42, min_value=0)
            
            with col_b:
                algorithm = st.selectbox(
                    "Algorithm",
                    ["Random Forest", "XGBoost", "Logistic Regression", "SVM"]
                )
                cross_validation = st.checkbox("Use Cross Validation", value=True)
            
            if st.button("Start Training", type="primary"):
                if st.session_state.data is not None and st.session_state.column_mapping:
                    with st.spinner("Training model... This may take a few minutes."):
                        try:
                            trainer = ModelTrainer()
                            processor = DataProcessor()
                            
                            # Process data
                            processed_data = processor.process_data(
                                st.session_state.data, 
                                st.session_state.column_mapping
                            )
                            
                            if processed_data is not None:
                                # Train model
                                model, metrics = trainer.train_model(
                                    processed_data,
                                    algorithm=algorithm.lower().replace(' ', '_'),
                                    test_size=test_size,
                                    random_state=random_state,
                                    cross_validation=cross_validation
                                )
                                
                                st.session_state.trained_model = model
                                st.session_state.training_metrics = metrics
                                
                                st.success("✅ Model trained successfully!")
                                
                                # Display training metrics
                                st.subheader("Training Results")
                                col_x, col_y, col_z, col_w = st.columns(4)
                                
                                with col_x:
                                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                                with col_y:
                                    st.metric("Precision", f"{metrics['precision']:.3f}")
                                with col_z:
                                    st.metric("Recall", f"{metrics['recall']:.3f}")
                                with col_w:
                                    st.metric("F1-Score", f"{metrics['f1']:.3f}")
                                
                                # Save model
                                model_filename = f"trained_model_{algorithm.lower().replace(' ', '_')}.joblib"
                                joblib.dump(model, f"models/{model_filename}")
                                st.info(f"Model saved as {model_filename}")
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        st.subheader("Model Information")
        if st.session_state.trained_model is not None:
            st.success("✅ Model Ready")
            st.write("Current model loaded and ready for predictions.")
            
            if hasattr(st.session_state, 'training_metrics'):
                st.write("**Last Training Metrics:**")
                for metric, value in st.session_state.training_metrics.items():
                    st.write(f"- {metric.title()}: {value:.3f}")
        else:
            st.info("No model loaded. Please load pretrained model or train a new one.")

def synthetic_data_section():
    st.header("🔬 Synthetic Data Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Synthetic Fraud Data")
        
        # Generation parameters
        col_a, col_b = st.columns(2)
        
        with col_a:
            num_samples = st.number_input(
                "Number of Samples", 
                min_value=100, 
                max_value=10000, 
                value=1000, 
                step=100
            )
            fraud_ratio = st.slider(
                "Fraud Ratio", 
                min_value=0.01, 
                max_value=0.5, 
                value=0.1, 
                step=0.01,
                format="%.2f"
            )
        
        with col_b:
            dataset_type = st.selectbox(
                "Dataset Type",
                ["Credit Card", "Mobile Money", "Debit Card", "Mixed"]
            )
            include_anomalies = st.checkbox("Include Anomaly Features", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            amount_range = st.slider(
                "Transaction Amount Range ($)",
                min_value=1,
                max_value=10000,
                value=(10, 5000),
                step=10
            )
            
            seasonal_patterns = st.checkbox("Include Seasonal Patterns", value=True)
            correlation_features = st.checkbox("Add Correlated Features", value=True)
        
        if st.button("Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                try:
                    generator = SyntheticGenerator()
                    
                    synthetic_data = generator.generate_fraud_data(
                        num_samples=num_samples,
                        fraud_ratio=fraud_ratio,
                        dataset_type=dataset_type.lower().replace(' ', '_'),
                        amount_range=amount_range,
                        include_anomalies=include_anomalies,
                        seasonal_patterns=seasonal_patterns,
                        correlation_features=correlation_features
                    )
                    
                    st.success("✅ Synthetic data generated successfully!")
                    
                    # Display sample
                    st.subheader("Generated Data Preview")
                    st.dataframe(synthetic_data.head(10), use_container_width=True)
                    
                    # Quick stats
                    fraud_count = sum(synthetic_data['is_fraud'])
                    st.subheader("Generation Summary")
                    
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        st.metric("Total Samples", num_samples)
                    with col_y:
                        st.metric("Fraud Samples", fraud_count)
                    with col_z:
                        st.metric("Actual Fraud Ratio", f"{fraud_count/num_samples:.2%}")
                    
                    # Download option
                    csv_buffer = io.StringIO()
                    synthetic_data.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Synthetic Data",
                        data=csv_string,
                        file_name=f"synthetic_fraud_data_{dataset_type.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                    
                    # Option to use for training
                    if st.button("Use for Model Training", type="primary"):
                        # Convert to required format
                        st.session_state.data = synthetic_data
                        st.session_state.column_mapping = {
                            'amount': 'amount',
                            'fraud_label': 'is_fraud',
                            'timestamp': 'timestamp' if 'timestamp' in synthetic_data.columns else None,
                            'merchant': 'merchant_category' if 'merchant_category' in synthetic_data.columns else None
                        }
                        st.success("✅ Synthetic data loaded for training!")
                        st.info("Switch to Model Training section to train with this data.")
                
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
    
    with col2:
        st.subheader("Generator Info")
        st.info("""
        **Synthetic Data Features:**
        - Realistic transaction patterns
        - Fraud indicators based on research
        - Multiple dataset types supported
        - Configurable fraud ratios
        - Anomaly detection features
        - Seasonal and temporal patterns
        """)

def results_section():
    st.header("📈 Results & Metrics")
    
    if st.session_state.analysis_results is None and st.session_state.trained_model is None:
        st.warning("⚠️ No analysis results or trained model available. Please upload data and train a model first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🎯 Model Performance", "📈 Visualizations"])
    
    with tab1:
        if st.session_state.analysis_results is not None:
            st.subheader("Fraud Analysis Results")
            
            results = st.session_state.analysis_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            fraud_transactions = results['fraud_transactions']
            total_transactions = len(st.session_state.data) if st.session_state.data is not None else 0
            
            with col1:
                st.metric("Total Transactions", total_transactions)
            with col2:
                st.metric("Fraud Cases", len(fraud_transactions))
            with col3:
                fraud_rate = (len(fraud_transactions) / total_transactions * 100) if total_transactions > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                total_fraud_amount = fraud_transactions['amount'].sum() if len(fraud_transactions) > 0 else 0
                st.metric("Total Fraud Amount", f"${total_fraud_amount:,.2f}")
            
            # Fraud transactions list
            if len(fraud_transactions) > 0:
                st.subheader("Fraud Transactions")
                
                # Add filters
                col_a, col_b = st.columns(2)
                with col_a:
                    min_amount = st.number_input("Min Amount Filter", value=0.0, min_value=0.0)
                with col_b:
                    max_amount = st.number_input("Max Amount Filter", value=float(fraud_transactions['amount'].max()))
                
                filtered_fraud = fraud_transactions[
                    (fraud_transactions['amount'] >= min_amount) & 
                    (fraud_transactions['amount'] <= max_amount)
                ]
                
                st.dataframe(
                    filtered_fraud.sort_values('amount', ascending=False),
                    use_container_width=True
                )
                
                # Download filtered results
                csv_buffer = io.StringIO()
                filtered_fraud.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Fraud Transactions",
                    data=csv_string,
                    file_name="fraud_transactions.csv",
                    mime="text/csv"
                )
            else:
                st.info("No fraud transactions found in the dataset.")
        else:
            st.info("No data analysis results available.")
    
    with tab2:
        if hasattr(st.session_state, 'training_metrics'):
            st.subheader("Model Performance Metrics")
            
            metrics = st.session_state.training_metrics
            
            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.3f}",
                    help="Overall correctness of the model"
                )
            
            with col2:
                st.metric(
                    "Precision",
                    f"{metrics['precision']:.3f}",
                    help="True positive rate among predicted positives"
                )
            
            with col3:
                st.metric(
                    "Recall",
                    f"{metrics['recall']:.3f}",
                    help="True positive rate among actual positives"
                )
            
            with col4:
                st.metric(
                    "F1-Score",
                    f"{metrics['f1']:.3f}",
                    help="Harmonic mean of precision and recall"
                )
            
            # Additional metrics if available
            if 'auc' in metrics:
                st.metric("AUC-ROC", f"{metrics['auc']:.3f}", help="Area Under the ROC Curve")
            
            # Model predictions on current data
            if st.session_state.data is not None and st.session_state.trained_model is not None:
                st.subheader("Current Dataset Predictions")
                
                if st.button("Run Predictions on Current Data", type="primary"):
                    with st.spinner("Running predictions..."):
                        try:
                            processor = DataProcessor()
                            processed_data = processor.process_data(
                                st.session_state.data, 
                                st.session_state.column_mapping
                            )
                            
                            if processed_data is not None:
                                # Prepare features for prediction
                                feature_columns = [col for col in processed_data.columns if col != 'is_fraud']
                                X = processed_data[feature_columns]
                                
                                # Make predictions
                                predictions = st.session_state.trained_model.predict(X)
                                probabilities = st.session_state.trained_model.predict_proba(X)[:, 1]
                                
                                # Add predictions to dataframe
                                results_df = processed_data.copy()
                                results_df['predicted_fraud'] = predictions
                                results_df['fraud_probability'] = probabilities
                                
                                # Show high-risk transactions
                                high_risk = results_df[results_df['fraud_probability'] > 0.5].sort_values(
                                    'fraud_probability', ascending=False
                                )
                                
                                st.subheader("High-Risk Transactions")
                                if len(high_risk) > 0:
                                    st.dataframe(high_risk.head(20), use_container_width=True)
                                    
                                    # Download predictions
                                    csv_buffer = io.StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    csv_string = csv_buffer.getvalue()
                                    
                                    st.download_button(
                                        label="📥 Download Predictions",
                                        data=csv_string,
                                        file_name="fraud_predictions.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("No high-risk transactions detected.")
                        
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.info("No model performance metrics available. Train a model first.")
    
    with tab3:
        st.subheader("Data Visualizations")
        
        if st.session_state.data is not None and st.session_state.analysis_results is not None:
            visualizer = Visualizer()
            
            # Create visualizations
            try:
                processed_data = None
                if st.session_state.column_mapping:
                    processor = DataProcessor()
                    processed_data = processor.process_data(
                        st.session_state.data, 
                        st.session_state.column_mapping
                    )
                
                if processed_data is not None:
                    # Fraud distribution
                    fig1 = visualizer.plot_fraud_distribution(processed_data)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Amount distribution
                    fig2 = visualizer.plot_amount_distribution(processed_data)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Time series if timestamp available
                    if 'timestamp' in processed_data.columns:
                        fig3 = visualizer.plot_fraud_timeline(processed_data)
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Correlation heatmap
                    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        fig4 = visualizer.plot_correlation_heatmap(processed_data[numeric_cols])
                        st.plotly_chart(fig4, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Visualization error: {str(e)}")
        else:
            st.info("No data available for visualization. Upload and analyze data first.")

if __name__ == "__main__":
    main()
