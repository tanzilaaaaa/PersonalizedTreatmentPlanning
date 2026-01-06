"""
Model training module for the Personalized Treatment Recommendation System.
Implements multiple ML algorithms for treatment recommendation and outcome prediction.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training of multiple ML models."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize all ML models."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42, probability=True)
        }
        print("Models initialized successfully.")
    
    def train_single_model(self, model, X_train, y_train, model_name):
        """Train a single model and return cross-validation scores."""
        print(f"Training {model_name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Fit the model on full training data
        model.fit(X_train, y_train)
        
        print(f"{model_name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model, cv_scores
    
    def train_all_models(self, X_train, y_train):
        """Train all models and compare performance."""
        if not self.models:
            self.initialize_models()
        
        results = {}
        trained_models = {}
        
        print("Training all models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            trained_model, cv_scores = self.train_single_model(model, X_train, y_train, name)
            
            results[name] = {
                'model': trained_model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            trained_models[name] = trained_model
            
            # Track best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = trained_model
                self.best_model_name = name
        
        print("-" * 50)
        print(f"Best model: {self.best_model_name} with CV accuracy: {self.best_score:.4f}")
        
        return results, trained_models
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='Random Forest'):
        """Perform hyperparameter tuning for the specified model."""
        print(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'SVM':
            model = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_type}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model, filepath):
        """Save trained model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class OutcomePredictionTrainer(ModelTrainer):
    """Specialized trainer for outcome prediction task."""
    
    def __init__(self):
        super().__init__()
        self.task_type = "outcome_prediction"
    
    def prepare_outcome_data(self, df, target_column='outcome'):
        """Prepare data specifically for outcome prediction."""
        # Features for outcome prediction (exclude treatment recommendation)
        feature_columns = [col for col in df.columns 
                          if col not in [target_column, 'patient_id', 'recommended_treatment']]
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y


class TreatmentRecommendationTrainer(ModelTrainer):
    """Specialized trainer for treatment recommendation task."""
    
    def __init__(self):
        super().__init__()
        self.task_type = "treatment_recommendation"
    
    def prepare_treatment_data(self, df, target_column='recommended_treatment'):
        """Prepare data specifically for treatment recommendation."""
        # Features for treatment recommendation (exclude outcome)
        feature_columns = [col for col in df.columns 
                          if col not in [target_column, 'patient_id', 'outcome']]
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y


if __name__ == "__main__":
    # Test model training with sample data
    from data_preprocessing import generate_sample_data, DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Generate and prepare data
    print("Generating sample data...")
    sample_data = generate_sample_data(1000)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(sample_data)
    
    # Encode categorical features
    categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
    encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(encoded_data)
    
    # Train outcome prediction model
    print("\n" + "="*60)
    print("TRAINING OUTCOME PREDICTION MODEL")
    print("="*60)
    
    outcome_trainer = OutcomePredictionTrainer()
    X_outcome, y_outcome = outcome_trainer.prepare_outcome_data(engineered_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_outcome, y_outcome)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train models
    outcome_results, outcome_models = outcome_trainer.train_all_models(X_train_scaled, y_train)
    
    # Save best outcome model
    outcome_trainer.save_model(outcome_trainer.best_model, 'models/outcome_prediction_model.pkl')
    
    print("\n" + "="*60)
    print("TRAINING TREATMENT RECOMMENDATION MODEL")
    print("="*60)
    
    # Train treatment recommendation model
    treatment_trainer = TreatmentRecommendationTrainer()
    X_treatment, y_treatment = treatment_trainer.prepare_treatment_data(engineered_data)
    X_train_t, X_test_t, y_train_t, y_test_t = preprocessor.split_data(X_treatment, y_treatment)
    
    # Scale features
    X_train_t_scaled, X_test_t_scaled = preprocessor.scale_features(X_train_t, X_test_t)
    
    # Train models
    treatment_results, treatment_models = treatment_trainer.train_all_models(X_train_t_scaled, y_train_t)
    
    # Save best treatment model
    treatment_trainer.save_model(treatment_trainer.best_model, 'models/treatment_recommendation_model.pkl')