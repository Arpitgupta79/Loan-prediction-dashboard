import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

# Main title
st.title("🏦 Loan Approval Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Prediction"])

# Data upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if page == "Data Analysis":
        st.header("📊 Data Analysis")
        
        # Data overview
        st.subheader("Data Overview")
        st.write(df.head())
        
        # Data statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            buffer = df.info(buf=None, max_cols=None, memory_usage=None, show_counts=None)
            st.text(str(buffer))
        
        with col2:
            st.subheader("Statistical Summary")
            st.write(df.describe())
        
        # Data visualization
        st.subheader("Data Visualization")
        
        # Correlation matrix
        st.plotly_chart(st.session_state.visualizer.plot_correlation_matrix(df))
        
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        # Data preprocessing
        with st.spinner("Preprocessing data..."):
            X_train, X_test, y_train, y_test = st.session_state.data_processor.prepare_data(df)
        
        # Model selection
        st.subheader("Model Selection")
        
        # Train models
        with st.spinner("Training models..."):
            cv_results = st.session_state.model_trainer.train_models(X_train, y_train)
            st.plotly_chart(st.session_state.visualizer.plot_metrics_comparison(cv_results))
        
        # Select best model
        selected_model = st.selectbox("Select model for final training",
                                    list(cv_results.keys()),
                                    format_func=lambda x: f"{x} (CV Score: {cv_results[x]['mean_cv_score']:.3f})")
        
        if st.button("Train Selected Model"):
            with st.spinner("Training final model..."):
                # Train final model
                model = st.session_state.model_trainer.train_best_model(X_train, y_train, selected_model)
                
                # Evaluate model
                metrics = st.session_state.model_trainer.evaluate_model(X_test, y_test)
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                col2.metric("Precision", f"{metrics['precision']:.3f}")
                col3.metric("Recall", f"{metrics['recall']:.3f}")
                col4.metric("F1 Score", f"{metrics['f1']:.3f}")
                col5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                
                # Feature importance
                if st.session_state.model_trainer.feature_importance is not None:
                    st.plotly_chart(st.session_state.visualizer.plot_feature_importance(
                        st.session_state.model_trainer.feature_importance))
                
                # Confusion matrix
                y_pred = model.predict(X_test)
                st.plotly_chart(st.session_state.visualizer.plot_confusion_matrix(y_test, y_pred))
                
    elif page == "Prediction":
        st.header("🎯 Prediction")
        
        # Input form for prediction
        st.subheader("Enter Loan Application Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            
        with col2:
            loan_term = st.number_input("Loan Term (months)", min_value=0)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
            residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
            commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
            luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
            bank_asset_value = st.number_input("Bank Asset Value", min_value=0)
            
        if st.button("Predict"):
            if st.session_state.model_trainer.best_model is None:
                st.error("Please train a model first!")
            else:
                # Prepare input data
                input_data = pd.DataFrame({
                    'no_of_dependents': [no_of_dependents],
                    'education': [education],
                    'self_employed': [self_employed],
                    'income_annum': [income_annum],
                    'loan_amount': [loan_amount],
                    'loan_term': [loan_term],
                    'cibil_score': [cibil_score],
                    'residential_assets_value': [residential_assets_value],
                    'commercial_assets_value': [commercial_assets_value],
                    'luxury_assets_value': [luxury_assets_value],
                    'bank_asset_value': [bank_asset_value]
                })
                
                # Process input data
                input_data = st.session_state.data_processor.engineer_features(input_data)
                input_data = st.session_state.data_processor.encode_categorical(input_data)
                
                # Make prediction
                prediction = st.session_state.model_trainer.predict(input_data)
                
                # Display result
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.success("Loan Approved! ✅")
                else:
                    st.error("Loan Rejected! ❌")
else:
    st.info("Please upload a CSV file to begin.")
