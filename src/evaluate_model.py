"""
Model evaluation module for the Personalized Treatment Recommendation System.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive evaluation of classification model."""
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("-" * 40)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_classification_report(self, y_true, y_pred, labels=None, title="Classification Report"):
        """Generate and display classification report."""
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        print(f"\n{title}")
        print("=" * len(title))
        print(classification_report(y_true, y_pred, target_names=labels))
        
        return df_report
    
    def plot_model_comparison(self, results_dict):
        """Compare multiple models performance."""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data for plotting
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': results_dict[model][metric]
                })
        
        df_comparison = pd.DataFrame(data)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_comparison, x='Model', y='Score', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return df_comparison
    
    def plot_learning_curves(self, model, X, y, model_name="Model"):
        """Plot learning curves to analyze model performance vs training size."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", top_n=15):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame for easier handling
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = feature_imp_df.head(top_n)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return feature_imp_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def generate_evaluation_report(self, model, X_test, y_test, feature_names, 
                                 class_labels, model_name="Model"):
        """Generate comprehensive evaluation report."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION REPORT - {model_name}")
        print(f"{'='*60}")
        
        # Basic evaluation
        results = self.evaluate_classification_model(model, X_test, y_test, model_name)
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = self.plot_confusion_matrix(results['y_true'], results['y_pred'], 
                                      class_labels, f"Confusion Matrix - {model_name}")
        
        # Classification report
        self.plot_classification_report(results['y_true'], results['y_pred'], 
                                      class_labels, f"Classification Report - {model_name}")
        
        # Feature importance (if available)
        feature_imp = self.plot_feature_importance(model, feature_names, model_name)
        
        return results, cm, feature_imp


class TreatmentEvaluator(ModelEvaluator):
    """Specialized evaluator for treatment recommendation models."""
    
    def evaluate_treatment_recommendations(self, model, X_test, y_test, 
                                         treatment_labels, model_name="Treatment Model"):
        """Evaluate treatment recommendation model with domain-specific metrics."""
        results = self.evaluate_classification_model(model, X_test, y_test, model_name)
        
        # Additional treatment-specific analysis
        self.analyze_treatment_patterns(results['y_true'], results['y_pred'], treatment_labels)
        
        return results
    
    def analyze_treatment_patterns(self, y_true, y_pred, treatment_labels):
        """Analyze treatment recommendation patterns."""
        print("\nTreatment Recommendation Analysis:")
        print("-" * 40)
        
        # Treatment distribution in predictions
        pred_counts = pd.Series(y_pred).value_counts()
        true_counts = pd.Series(y_true).value_counts()
        
        print("Treatment Distribution:")
        comparison_df = pd.DataFrame({
            'True Distribution': true_counts,
            'Predicted Distribution': pred_counts
        }).fillna(0)
        
        print(comparison_df)
        
        # Plot treatment distribution comparison
        plt.figure(figsize=(12, 6))
        comparison_df.plot(kind='bar')
        plt.title('Treatment Distribution: True vs Predicted')
        plt.xlabel('Treatment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()


class OutcomeEvaluator(ModelEvaluator):
    """Specialized evaluator for outcome prediction models."""
    
    def evaluate_outcome_predictions(self, model, X_test, y_test, 
                                   outcome_labels, model_name="Outcome Model"):
        """Evaluate outcome prediction model with domain-specific metrics."""
        results = self.evaluate_classification_model(model, X_test, y_test, model_name)
        
        # Additional outcome-specific analysis
        self.analyze_outcome_patterns(results['y_true'], results['y_pred'], outcome_labels)
        
        return results
    
    def analyze_outcome_patterns(self, y_true, y_pred, outcome_labels):
        """Analyze outcome prediction patterns."""
        print("\nOutcome Prediction Analysis:")
        print("-" * 40)
        
        # Calculate outcome-specific metrics
        for i, label in enumerate(outcome_labels):
            true_positive = np.sum((y_true == i) & (y_pred == i))
            false_positive = np.sum((y_true != i) & (y_pred == i))
            false_negative = np.sum((y_true == i) & (y_pred != i))
            
            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0
                
            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0
            
            print(f"{label}: Precision={precision:.3f}, Recall={recall:.3f}")


if __name__ == "__main__":
    # Test evaluation with sample data
    from data_preprocessing import generate_sample_data, DataPreprocessor
    from feature_engineering import FeatureEngineer
    from train_model import OutcomePredictionTrainer, TreatmentRecommendationTrainer
    
    print("Testing model evaluation...")
    
    # Generate and prepare data
    sample_data = generate_sample_data(500)
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(sample_data)
    
    # Encode and engineer features
    categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
    encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
    
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.engineer_all_features(encoded_data)
    
    # Test outcome prediction evaluation
    outcome_trainer = OutcomePredictionTrainer()
    outcome_trainer.initialize_models()
    
    X_outcome, y_outcome = outcome_trainer.prepare_outcome_data(engineered_data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_outcome, y_outcome)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train a simple model for testing
    from sklearn.ensemble import RandomForestClassifier
    test_model = RandomForestClassifier(random_state=42, n_estimators=50)
    test_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    evaluator = OutcomeEvaluator()
    outcome_labels = ['Improved', 'Not Improved', 'Stable']  # Adjust based on your data
    feature_names = X_outcome.columns.tolist()
    
    results, cm, feature_imp = evaluator.generate_evaluation_report(
        test_model, X_test_scaled, y_test, feature_names, outcome_labels, "Test Random Forest"
    )