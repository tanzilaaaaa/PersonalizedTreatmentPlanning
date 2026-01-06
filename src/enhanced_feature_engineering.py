"""
Enhanced Feature Engineering module with advanced techniques for better model performance.
Creates robust features that improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """Enhanced feature engineering with advanced techniques."""
    
    def __init__(self):
        self.polynomial_features = None
        self.pca = None
        self.kmeans = None
        self.feature_stats = {}
        
    def create_age_based_features(self, df):
        """Create comprehensive age-based features."""
        df = df.copy()
        
        # Age groups with more granularity
        df['age_group_detailed'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 65, 75, 100], 
            labels=['very_young', 'young', 'young_adult', 'middle_aged', 'mature', 'senior', 'elderly']
        )
        
        # Age risk categories
        df['age_risk_category'] = pd.cut(
            df['age'],
            bins=[0, 40, 60, 100],
            labels=['low_risk', 'medium_risk', 'high_risk']
        )
        
        # Age squared (non-linear relationship)
        df['age_squared'] = df['age'] ** 2
        
        # Age normalized (0-1 scale)
        df['age_normalized'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        
        # Age deviation from mean
        df['age_deviation'] = df['age'] - df['age'].mean()
        
        return df
    
    def create_severity_features(self, df):
        """Create enhanced severity-based features."""
        df = df.copy()
        
        # Severity score (robust mapping)
        severity_mapping = {0: 1, 1: 2, 2: 3}  # Assuming 0=Mild, 1=Moderate, 2=Severe
        df['severity_score'] = df['severity'].map(severity_mapping).fillna(2)  # Default to moderate
        
        # Severity binary flags
        df['is_severe'] = (df['severity'] == 2).astype(int)
        df['is_mild'] = (df['severity'] == 0).astype(int)
        df['is_moderate'] = (df['severity'] == 1).astype(int)
        
        # Severity risk multiplier
        df['severity_risk_multiplier'] = df['severity_score'] * 1.5
        
        return df
    
    def create_treatment_history_features(self, df):
        """Create comprehensive treatment history features."""
        df = df.copy()
        
        # Basic treatment history flag
        df['has_previous_treatment'] = (df['previous_treatment'] != 0).astype(int)
        
        # Treatment complexity score
        treatment_complexity = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3}  # Assuming different treatment types
        df['treatment_complexity'] = df['previous_treatment'].map(treatment_complexity).fillna(0)
        
        # Treatment failure indicator (if had previous treatment but still needs new treatment)
        df['potential_treatment_failure'] = df['has_previous_treatment']
        
        return df
    
    def create_interaction_features(self, df):
        """Create meaningful interaction features."""
        df = df.copy()
        
        # Age-severity interactions
        if 'age' in df.columns and 'severity_score' in df.columns:
            df['age_severity_interaction'] = df['age'] * df['severity_score']
            df['age_severity_ratio'] = df['age'] / (df['severity_score'] + 1)  # +1 to avoid division by zero
        
        # Gender-age interactions
        if 'gender' in df.columns and 'age' in df.columns:
            df['gender_age_interaction'] = df['gender'] * df['age']
            df['is_elderly_female'] = ((df['gender'] == 0) & (df['age'] > 65)).astype(int)
            df['is_elderly_male'] = ((df['gender'] == 1) & (df['age'] > 65)).astype(int)
        
        # Diagnosis-severity interactions
        if 'diagnosis' in df.columns and 'severity_score' in df.columns:
            df['diagnosis_severity_interaction'] = df['diagnosis'] * df['severity_score']
        
        # Treatment history-severity interactions
        if 'has_previous_treatment' in df.columns and 'severity_score' in df.columns:
            df['treatment_severity_interaction'] = df['has_previous_treatment'] * df['severity_score']
        
        return df
    
    def create_risk_assessment_features(self, df):
        """Create comprehensive risk assessment features."""
        df = df.copy()
        
        # Initialize risk score
        df['comprehensive_risk_score'] = 0.0
        
        # Age risk component
        if 'age' in df.columns:
            df['age_risk'] = np.where(df['age'] > 65, 3, 
                                    np.where(df['age'] > 50, 2, 1))
            df['comprehensive_risk_score'] += df['age_risk'] * 0.3
        
        # Severity risk component
        if 'severity_score' in df.columns:
            df['comprehensive_risk_score'] += df['severity_score'] * 0.4
        
        # Treatment history risk component
        if 'has_previous_treatment' in df.columns:
            df['comprehensive_risk_score'] += df['has_previous_treatment'] * 0.2
        
        # Gender risk component (example: certain conditions more common in specific genders)
        if 'gender' in df.columns:
            df['comprehensive_risk_score'] += df['gender'] * 0.1
        
        # Risk categories
        df['risk_category'] = pd.cut(
            df['comprehensive_risk_score'],
            bins=[0, 1.5, 2.5, 5.0],
            labels=['low', 'medium', 'high']
        )
        
        return df
    
    def create_polynomial_features(self, df, degree=2, include_columns=None):
        """Create polynomial features for numerical columns."""
        df = df.copy()
        
        if include_columns is None:
            # Select numerical columns for polynomial features
            numerical_cols = ['age', 'severity_score']
            include_columns = [col for col in numerical_cols if col in df.columns]
        
        if len(include_columns) < 2:
            return df
        
        # Create polynomial features
        poly_data = df[include_columns].fillna(df[include_columns].median())
        
        if self.polynomial_features is None:
            self.polynomial_features = PolynomialFeatures(
                degree=degree, 
                include_bias=False, 
                interaction_only=True
            )
            poly_features = self.polynomial_features.fit_transform(poly_data)
        else:
            poly_features = self.polynomial_features.transform(poly_data)
        
        # Get feature names
        feature_names = self.polynomial_features.get_feature_names_out(include_columns)
        
        # Add polynomial features to dataframe
        for i, name in enumerate(feature_names):
            if name not in include_columns:  # Don't duplicate original features
                df[f'poly_{name}'] = poly_features[:, i]
        
        return df
    
    def create_clustering_features(self, df, n_clusters=5):
        """Create clustering-based features."""
        df = df.copy()
        
        # Select features for clustering
        cluster_features = ['age', 'severity_score', 'has_previous_treatment']
        available_features = [col for col in cluster_features if col in df.columns]
        
        if len(available_features) < 2:
            return df
        
        cluster_data = df[available_features].fillna(df[available_features].median())
        
        # Standardize data for clustering
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform clustering
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(cluster_data_scaled)
        else:
            cluster_labels = self.kmeans.predict(cluster_data_scaled)
        
        df['patient_cluster'] = cluster_labels
        
        # Create cluster-based features
        for i in range(n_clusters):
            df[f'is_cluster_{i}'] = (cluster_labels == i).astype(int)
        
        # Distance to cluster centers
        distances = self.kmeans.transform(cluster_data_scaled)
        df['min_cluster_distance'] = np.min(distances, axis=1)
        df['cluster_distance_std'] = np.std(distances, axis=1)
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features based on data distribution."""
        df = df.copy()
        
        # Z-scores for numerical features
        numerical_cols = ['age', 'severity_score']
        
        for col in numerical_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean_val) / (std_val + 1e-8)  # Add small value to avoid division by zero
                df[f'{col}_is_outlier'] = (np.abs(df[f'{col}_zscore']) > 2).astype(int)
        
        return df
    
    def create_domain_specific_features(self, df):
        """Create healthcare domain-specific features."""
        df = df.copy()
        
        # Comorbidity risk (based on age and previous treatments)
        if 'age' in df.columns and 'has_previous_treatment' in df.columns:
            df['comorbidity_risk'] = (
                (df['age'] > 60).astype(int) * 2 + 
                df['has_previous_treatment'] * 1.5
            )
        
        # Treatment urgency score
        if 'severity_score' in df.columns and 'age' in df.columns:
            df['treatment_urgency'] = (
                df['severity_score'] * 0.6 + 
                (df['age'] > 70).astype(int) * 0.4
            )
        
        # Recovery likelihood (inverse of risk factors)
        if 'comprehensive_risk_score' in df.columns:
            max_risk = df['comprehensive_risk_score'].max()
            df['recovery_likelihood'] = (max_risk - df['comprehensive_risk_score']) / max_risk
        
        return df
    
    def engineer_all_features(self, df, include_polynomial=True, include_clustering=True):
        """Apply all enhanced feature engineering techniques."""
        print("Starting enhanced feature engineering...")
        
        original_shape = df.shape
        
        # Step 1: Age-based features
        df = self.create_age_based_features(df)
        
        # Step 2: Severity features
        df = self.create_severity_features(df)
        
        # Step 3: Treatment history features
        df = self.create_treatment_history_features(df)
        
        # Step 4: Interaction features
        df = self.create_interaction_features(df)
        
        # Step 5: Risk assessment features
        df = self.create_risk_assessment_features(df)
        
        # Step 6: Statistical features
        df = self.create_statistical_features(df)
        
        # Step 7: Domain-specific features
        df = self.create_domain_specific_features(df)
        
        # Step 8: Polynomial features (optional)
        if include_polynomial:
            df = self.create_polynomial_features(df)
        
        # Step 9: Clustering features (optional)
        if include_clustering:
            df = self.create_clustering_features(df)
        
        # Handle any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical NaN values
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        print(f"Enhanced feature engineering completed.")
        print(f"Original shape: {original_shape}")
        print(f"New shape: {df.shape}")
        print(f"Features added: {df.shape[1] - original_shape[1]}")
        
        # Store feature statistics
        self.feature_stats = {
            'original_features': original_shape[1],
            'final_features': df.shape[1],
            'features_added': df.shape[1] - original_shape[1],
            'feature_names': list(df.columns)
        }
        
        return df
    
    def get_feature_importance_groups(self):
        """Get feature groups for importance analysis."""
        if not self.feature_stats:
            return {}
        
        feature_groups = {
            'age_features': [col for col in self.feature_stats['feature_names'] if 'age' in col.lower()],
            'severity_features': [col for col in self.feature_stats['feature_names'] if 'severity' in col.lower()],
            'treatment_features': [col for col in self.feature_stats['feature_names'] if 'treatment' in col.lower()],
            'interaction_features': [col for col in self.feature_stats['feature_names'] if 'interaction' in col.lower()],
            'risk_features': [col for col in self.feature_stats['feature_names'] if 'risk' in col.lower()],
            'cluster_features': [col for col in self.feature_stats['feature_names'] if 'cluster' in col.lower()],
            'polynomial_features': [col for col in self.feature_stats['feature_names'] if 'poly' in col.lower()],
            'statistical_features': [col for col in self.feature_stats['feature_names'] if any(stat in col.lower() for stat in ['zscore', 'outlier', 'std'])]
        }
        
        return feature_groups


def run_enhanced_feature_engineering_demo():
    """Demonstrate enhanced feature engineering capabilities."""
    print("🔧 ENHANCED FEATURE ENGINEERING DEMO")
    print("=" * 60)
    
    # Import required modules
    from data_preprocessing import generate_sample_data, DataPreprocessor
    
    # Generate sample data
    print("Generating sample data...")
    sample_data = generate_sample_data(1000)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(sample_data)
    
    # Encode categorical features
    categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
    encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
    
    print(f"Original data shape: {encoded_data.shape}")
    print(f"Original columns: {list(encoded_data.columns)}")
    
    # Apply enhanced feature engineering
    enhanced_engineer = EnhancedFeatureEngineer()
    enhanced_data = enhanced_engineer.engineer_all_features(encoded_data)
    
    print(f"\nEnhanced data shape: {enhanced_data.shape}")
    print(f"Features added: {enhanced_data.shape[1] - encoded_data.shape[1]}")
    
    # Show feature groups
    feature_groups = enhanced_engineer.get_feature_importance_groups()
    print(f"\nFeature groups created:")
    for group_name, features in feature_groups.items():
        if features:
            print(f"  {group_name}: {len(features)} features")
    
    # Show sample of new features
    new_features = [col for col in enhanced_data.columns if col not in encoded_data.columns]
    print(f"\nSample of new features: {new_features[:10]}")
    
    return enhanced_data, enhanced_engineer


if __name__ == "__main__":
    enhanced_data, engineer = run_enhanced_feature_engineering_demo()