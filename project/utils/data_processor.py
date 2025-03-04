import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}

    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Clean column names - remove leading/trailing spaces
        df.columns = df.columns.str.strip()

        # Remove rows with missing values
        df = df.dropna()

        # Remove duplicates
        df = df.drop_duplicates()

        # Convert currency values to numerical
        numeric_columns = ['income_annum', 'loan_amount', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def engineer_features(self, df):
        """Create new features from existing data"""
        # Calculate debt-to-income ratio
        df['debt_to_income'] = df['loan_amount'] / df['income_annum']

        # Calculate total assets
        df['total_assets'] = (df['residential_assets_value'] + 
                            df['commercial_assets_value'] + 
                            df['luxury_assets_value'] + 
                            df['bank_asset_value'])

        # Calculate loan to asset ratio
        df['loan_to_asset'] = df['loan_amount'] / df['total_assets']

        return df

    def encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_columns = ['education', 'self_employed']

        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column].str.strip())

        return df

    def prepare_data(self, df):
        """Prepare data for model training"""
        # Clean data
        df = self.clean_data(df)

        # Engineer features
        df = self.engineer_features(df)

        # Encode categorical variables
        df = self.encode_categorical(df)

        # Prepare features and target
        X = df.drop(['loan_id', 'loan_status'], axis=1)
        y = (df['loan_status'].str.strip() == 'Approved').astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test