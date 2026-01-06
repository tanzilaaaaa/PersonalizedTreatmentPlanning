"""
Data preprocessing module for the Personalized Treatment Recommendation System.
Handles data cleaning, missing values, and basic transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Handles data preprocessing tasks including cleaning and encoding."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return None
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and duplicates."""
        print("Starting data cleaning...")
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
        
        # Fill missing numerical values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("Data cleaning completed.")
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical features using Label Encoder."""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
        
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features using MinMaxScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_features_target(self, df, target_column, feature_columns=None):
        """Prepare features and target variables."""
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_sample_data(n_samples=1000, filepath='data/raw/sample_patient_data.csv'):
    """Generate sample patient data for demonstration."""
    np.random.seed(42)
    
    # Generate patient data
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'symptoms': np.random.choice(['Fever', 'Cough', 'Fatigue', 'Headache', 'Nausea'], n_samples),
        'diagnosis': np.random.choice(['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Depression'], n_samples),
        'previous_treatment': np.random.choice(['None', 'Medication A', 'Medication B', 'Therapy', 'Surgery'], n_samples),
        'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], n_samples),
        'recommended_treatment': np.random.choice(['Treatment A', 'Treatment B', 'Treatment C', 'Treatment D'], n_samples),
        'outcome': np.random.choice(['Improved', 'Stable', 'Not Improved'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Sample data generated and saved to {filepath}")
    return df


if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_sample_data()
    print(sample_data.head())
    print(f"\nDataset info:")
    print(sample_data.info())