"""
Feature engineering module for the Personalized Treatment Recommendation System.
Creates new features and transforms existing ones for better model performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class FeatureEngineer:
    """Handles feature engineering tasks."""
    
    def __init__(self):
        self.one_hot_encoders = {}
        
    def create_age_groups(self, df, age_column='age'):
        """Create age group categories."""
        df = df.copy()
        df['age_group'] = pd.cut(df[age_column], 
                                bins=[0, 30, 50, 70, 100], 
                                labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
        return df
    
    def create_severity_score(self, df, severity_column='severity'):
        """Convert severity to numerical score."""
        df = df.copy()
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        df['severity_score'] = df[severity_column].map(severity_mapping)
        return df
    
    def create_treatment_history_flag(self, df, previous_treatment_column='previous_treatment'):
        """Create binary flag for treatment history."""
        df = df.copy()
        df['has_previous_treatment'] = (df[previous_treatment_column] != 'None').astype(int)
        return df
    
    def one_hot_encode_features(self, df, categorical_columns):
        """Apply one-hot encoding to categorical features."""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                # Get unique values
                unique_values = df_encoded[col].unique()
                
                # Create one-hot encoded columns
                for value in unique_values:
                    new_col_name = f"{col}_{value}"
                    df_encoded[new_col_name] = (df_encoded[col] == value).astype(int)
                
                print(f"One-hot encoded column: {col}")
        
        return df_encoded
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables."""
        df = df.copy()
        
        # Age-severity interaction
        if 'age' in df.columns and 'severity_score' in df.columns:
            # Fill NaN values before creating interaction
            severity_median = df['severity_score'].median()
            df['severity_score'] = df['severity_score'].fillna(severity_median)
            df['age_severity_interaction'] = df['age'] * df['severity_score']
        
        # Gender-age interaction (if gender is encoded)
        if 'gender' in df.columns and 'age' in df.columns:
            # Assuming gender is already label encoded (0, 1)
            df['gender_age_interaction'] = df['gender'] * df['age']
        
        return df
    
    def create_risk_score(self, df):
        """Create a composite risk score based on multiple factors."""
        df = df.copy()
        
        # Initialize risk score
        df['risk_score'] = 0.0
        
        # Age contribution (higher age = higher risk)
        if 'age' in df.columns:
            df['risk_score'] += (df['age'] / 100) * 30
        
        # Severity contribution
        if 'severity_score' in df.columns:
            # Fill NaN values with median severity score
            severity_median = df['severity_score'].median()
            df['severity_score'] = df['severity_score'].fillna(severity_median)
            df['risk_score'] += df['severity_score'] * 20
        
        # Previous treatment contribution (having previous treatment = higher risk)
        if 'has_previous_treatment' in df.columns:
            df['risk_score'] += df['has_previous_treatment'] * 15
        
        return df
    
    def engineer_all_features(self, df):
        """Apply all feature engineering steps."""
        print("Starting feature engineering...")
        
        # Create age groups
        df = self.create_age_groups(df)
        
        # Create severity score
        df = self.create_severity_score(df)
        
        # Create treatment history flag
        df = self.create_treatment_history_flag(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create risk score
        df = self.create_risk_score(df)
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        print("Feature engineering completed.")
        print(f"New dataset shape: {df.shape}")
        
        return df
    
    def get_feature_importance_data(self, df):
        """Prepare data for feature importance analysis."""
        # Select numerical features for correlation analysis
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns
        numerical_features = [col for col in numerical_features if 'id' not in col.lower()]
        
        return df[numerical_features]


if __name__ == "__main__":
    # Test feature engineering with sample data
    from data_preprocessing import generate_sample_data, DataPreprocessor
    
    # Generate sample data
    sample_data = generate_sample_data(500)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(sample_data)
    
    # Encode categorical features
    categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
    encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(encoded_data)
    
    print("\nEngineered features:")
    print(engineered_data.columns.tolist())
    print("\nSample of engineered data:")
    print(engineered_data.head())