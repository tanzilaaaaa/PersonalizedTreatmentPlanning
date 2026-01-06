"""
Advanced Model Training module for improved performance.
Implements hyperparameter tuning, ensemble methods, and advanced preprocessing.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')


class AdvancedModelTrainer:
    """Advanced model trainer with hyperparameter tuning and ensemble methods."""
    
    def __init__(self):
        self.best_models = {}
        self.ensemble_model = None
        self.best_score = 0
        self.best_model_name = None
        
    def get_optimized_models(self):
        """Get models with optimized hyperparameters."""
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'degree': [2, 3, 4]
                }
            }
        }
        return models
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest', 
                            search_type='grid', cv_folds=5, n_jobs=-1):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to tune
            search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        models = self.get_optimized_models()
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(models.keys())}")
        
        model_config = models[model_name]
        model = model_config['model']
        param_grid = model_config['params']
        
        print(f"Tuning hyperparameters for {model_name}...")
        print(f"Search space size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Use stratified k-fold for better cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy',
                n_jobs=n_jobs, verbose=1, return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring='accuracy',
                n_iter=50, n_jobs=n_jobs, verbose=1, 
                random_state=42, return_train_score=True
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")
        print(f"Best CV std: {search.cv_results_['std_test_score'][search.best_index_]:.4f}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def tune_all_models(self, X_train, y_train, search_type='random', cv_folds=5):
        """Tune hyperparameters for all models."""
        print("=" * 60)
        print("HYPERPARAMETER TUNING FOR ALL MODELS")
        print("=" * 60)
        
        tuned_models = {}
        results = {}
        
        models = self.get_optimized_models()
        
        for model_name in models.keys():
            print(f"\n{'-' * 40}")
            print(f"Tuning {model_name}")
            print(f"{'-' * 40}")
            
            try:
                best_model, best_score, best_params = self.hyperparameter_tuning(
                    X_train, y_train, model_name, search_type, cv_folds
                )
                
                tuned_models[model_name] = best_model
                results[model_name] = {
                    'best_score': best_score,
                    'best_params': best_params,
                    'model': best_model
                }
                
                # Track overall best model
                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"Error tuning {model_name}: {e}")
                continue
        
        self.best_models = tuned_models
        
        print(f"\n{'=' * 60}")
        print("HYPERPARAMETER TUNING RESULTS")
        print(f"{'=' * 60}")
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['best_score']:.4f}")
        
        print(f"\nBest overall model: {self.best_model_name} ({self.best_score:.4f})")
        
        return results
    
    def create_ensemble_model(self, X_train, y_train, top_n=3):
        """Create an ensemble model from the best performing models."""
        if not self.best_models:
            raise ValueError("No tuned models available. Run tune_all_models first.")
        
        print(f"\nCreating ensemble model with top {top_n} models...")
        
        # Sort models by performance
        model_scores = []
        for name, model in self.best_models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model_scores.append((name, model, cv_scores.mean()))
        
        # Select top N models
        model_scores.sort(key=lambda x: x[2], reverse=True)
        top_models = model_scores[:top_n]
        
        print("Selected models for ensemble:")
        for name, model, score in top_models:
            print(f"  {name}: {score:.4f}")
        
        # Create voting classifier
        estimators = [(name, model) for name, model, _ in top_models]
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability-based voting
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_cv_scores = cross_val_score(
            self.ensemble_model, X_train, y_train, cv=5, scoring='accuracy'
        )
        ensemble_score = ensemble_cv_scores.mean()
        
        print(f"\nEnsemble CV score: {ensemble_score:.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        
        # Update best model if ensemble is better
        if ensemble_score > self.best_score:
            self.best_score = ensemble_score
            self.best_model_name = "Ensemble"
        
        return self.ensemble_model, ensemble_score
    
    def feature_selection_optimization(self, X_train, y_train, X_test, y_test, 
                                     selection_methods=['univariate', 'rfe']):
        """Optimize feature selection for better performance."""
        print("\n" + "=" * 60)
        print("FEATURE SELECTION OPTIMIZATION")
        print("=" * 60)
        
        results = {}
        
        # Baseline performance (all features)
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_cv = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='accuracy')
        baseline_score = baseline_cv.mean()
        
        print(f"Baseline (all {X_train.shape[1]} features): {baseline_score:.4f}")
        results['baseline'] = {
            'n_features': X_train.shape[1],
            'cv_score': baseline_score,
            'features': list(X_train.columns) if hasattr(X_train, 'columns') else list(range(X_train.shape[1]))
        }
        
        if 'univariate' in selection_methods:
            print("\nTesting univariate feature selection...")
            
            # Test different numbers of features
            feature_counts = [5, 8, 10, 12, 15, min(20, X_train.shape[1])]
            
            for k in feature_counts:
                if k >= X_train.shape[1]:
                    continue
                    
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
                cv_score = cv_scores.mean()
                
                print(f"  Top {k} features: {cv_score:.4f}")
                
                results[f'univariate_{k}'] = {
                    'n_features': k,
                    'cv_score': cv_score,
                    'selector': selector
                }
        
        if 'rfe' in selection_methods:
            print("\nTesting recursive feature elimination...")
            
            feature_counts = [5, 8, 10, 12]
            
            for k in feature_counts:
                if k >= X_train.shape[1]:
                    continue
                
                base_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed
                selector = RFE(base_model, n_features_to_select=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
                cv_score = cv_scores.mean()
                
                print(f"  RFE {k} features: {cv_score:.4f}")
                
                results[f'rfe_{k}'] = {
                    'n_features': k,
                    'cv_score': cv_score,
                    'selector': selector
                }
        
        # Find best feature selection method
        best_method = max(results.keys(), key=lambda x: results[x]['cv_score'])
        best_score = results[best_method]['cv_score']
        
        print(f"\nBest feature selection: {best_method} with {best_score:.4f}")
        
        return results, best_method
    
    def advanced_preprocessing_optimization(self, X_train, y_train):
        """Test different preprocessing approaches."""
        print("\n" + "=" * 60)
        print("PREPROCESSING OPTIMIZATION")
        print("=" * 60)
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'No Scaling': None
        }
        
        results = {}
        
        for scaler_name, scaler in scalers.items():
            print(f"Testing {scaler_name}...")
            
            if scaler is not None:
                X_scaled = scaler.fit_transform(X_train)
            else:
                X_scaled = X_train
            
            # Test with Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring='accuracy')
            cv_score = cv_scores.mean()
            
            print(f"  {scaler_name}: {cv_score:.4f}")
            
            results[scaler_name] = {
                'cv_score': cv_score,
                'scaler': scaler
            }
        
        best_scaler = max(results.keys(), key=lambda x: results[x]['cv_score'])
        print(f"\nBest preprocessing: {best_scaler}")
        
        return results, best_scaler
    
    def comprehensive_optimization(self, X_train, y_train, X_test, y_test):
        """Run comprehensive model optimization pipeline."""
        print("🚀 COMPREHENSIVE MODEL OPTIMIZATION")
        print("=" * 80)
        
        optimization_results = {}
        
        # Step 1: Preprocessing optimization
        preprocessing_results, best_preprocessing = self.advanced_preprocessing_optimization(X_train, y_train)
        optimization_results['preprocessing'] = preprocessing_results
        
        # Apply best preprocessing
        best_scaler = preprocessing_results[best_preprocessing]['scaler']
        if best_scaler is not None:
            X_train_processed = best_scaler.fit_transform(X_train)
            X_test_processed = best_scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Step 2: Feature selection optimization
        feature_results, best_feature_method = self.feature_selection_optimization(
            X_train_processed, y_train, X_test_processed, y_test
        )
        optimization_results['feature_selection'] = feature_results
        
        # Step 3: Hyperparameter tuning
        tuning_results = self.tune_all_models(X_train_processed, y_train, search_type='random')
        optimization_results['hyperparameter_tuning'] = tuning_results
        
        # Step 4: Ensemble creation
        ensemble_model, ensemble_score = self.create_ensemble_model(X_train_processed, y_train)
        optimization_results['ensemble'] = {
            'model': ensemble_model,
            'cv_score': ensemble_score
        }
        
        # Final evaluation
        print("\n" + "=" * 80)
        print("FINAL OPTIMIZATION RESULTS")
        print("=" * 80)
        
        print(f"Best preprocessing: {best_preprocessing}")
        print(f"Best feature selection: {best_feature_method}")
        print(f"Best individual model: {self.best_model_name} ({self.best_score:.4f})")
        print(f"Ensemble model score: {ensemble_score:.4f}")
        
        # Test set evaluation
        if self.best_model_name == "Ensemble":
            final_model = ensemble_model
        else:
            final_model = self.best_models[self.best_model_name]
        
        test_score = final_model.score(X_test_processed, y_test)
        print(f"Final test set accuracy: {test_score:.4f}")
        
        return {
            'final_model': final_model,
            'test_score': test_score,
            'cv_score': max(self.best_score, ensemble_score),
            'preprocessing': best_scaler,
            'optimization_results': optimization_results
        }


def run_advanced_training_demo():
    """Run a demonstration of advanced model training."""
    print("🏥 ADVANCED MODEL TRAINING DEMO")
    print("=" * 60)
    
    # Import required modules
    from data_preprocessing import generate_sample_data, DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Generate larger dataset for better training
    print("Generating enhanced training dataset...")
    sample_data = generate_sample_data(3000)  # Larger dataset
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(sample_data)
    
    # Encode categorical features
    categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
    encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
    
    # Basic feature engineering (avoiding complex features that cause NaN)
    feature_engineer = FeatureEngineer()
    
    # Add only stable features
    engineered_data = encoded_data.copy()
    engineered_data = feature_engineer.create_treatment_history_flag(engineered_data)
    
    # Prepare data for training
    from train_model import OutcomePredictionTrainer
    trainer = OutcomePredictionTrainer()
    X, y = trainer.prepare_outcome_data(engineered_data)
    
    # Select numeric features only
    X_numeric = X.select_dtypes(include=[np.number])
    X_clean = X_numeric.fillna(X_numeric.median())  # Handle any remaining NaN
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_clean, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution: {dict(y_train.value_counts())}")
    
    # Run advanced training
    advanced_trainer = AdvancedModelTrainer()
    results = advanced_trainer.comprehensive_optimization(X_train, y_train, X_test, y_test)
    
    print("\n🎉 Advanced training completed!")
    print(f"Final model performance: {results['test_score']:.4f}")
    
    return results


if __name__ == "__main__":
    results = run_advanced_training_demo()