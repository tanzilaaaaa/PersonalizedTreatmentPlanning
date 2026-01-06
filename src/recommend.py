"""
Treatment recommendation module for the Personalized Treatment Recommendation System.
Provides treatment recommendations and outcome predictions for new patients.
"""

import pandas as pd
import numpy as np
import pickle
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer


class TreatmentRecommendationSystem:
    """Main system for generating treatment recommendations and outcome predictions."""
    
    def __init__(self):
        self.treatment_model = None
        self.outcome_model = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.treatment_labels = ['Treatment A', 'Treatment B', 'Treatment C', 'Treatment D']
        self.outcome_labels = ['Improved', 'Not Improved', 'Stable']
        
    def load_models(self, treatment_model_path, outcome_model_path):
        """Load trained models from files."""
        try:
            with open(treatment_model_path, 'rb') as f:
                self.treatment_model = pickle.load(f)
            print(f"Treatment model loaded from {treatment_model_path}")
            
            with open(outcome_model_path, 'rb') as f:
                self.outcome_model = pickle.load(f)
            print(f"Outcome model loaded from {outcome_model_path}")
            
            return True
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess new patient data for prediction."""
        # Convert to DataFrame if it's a dictionary
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Clean data
        df = self.preprocessor.clean_data(df)
        
        # Encode categorical features
        categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
        
        # Use existing label encoders if available
        for col in categorical_cols:
            if col in df.columns and col in self.preprocessor.label_encoders:
                le = self.preprocessor.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] 
                                      if str(x) in le.classes_ else 0)
            elif col in df.columns:
                # If no encoder exists, create a simple mapping
                unique_vals = df[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)
        
        # Apply feature engineering
        df = self.feature_engineer.engineer_all_features(df)
        
        return df
    
    def recommend_treatment(self, patient_data, return_probabilities=False):
        """Recommend treatment for a patient."""
        if self.treatment_model is None:
            raise ValueError("Treatment model not loaded. Please load models first.")
        
        # Preprocess patient data
        processed_data = self.preprocess_patient_data(patient_data)
        
        # Prepare features (exclude outcome and patient_id)
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['outcome', 'patient_id', 'recommended_treatment']]
        X = processed_data[feature_columns]
        
        # Scale features
        X_scaled = self.preprocessor.scaler.transform(X)
        
        # Make prediction
        treatment_pred = self.treatment_model.predict(X_scaled)
        
        # Get prediction probabilities if requested
        if return_probabilities and hasattr(self.treatment_model, 'predict_proba'):
            treatment_proba = self.treatment_model.predict_proba(X_scaled)
            return treatment_pred, treatment_proba
        
        return treatment_pred
    
    def predict_outcome(self, patient_data, recommended_treatment=None, return_probabilities=False):
        """Predict treatment outcome for a patient."""
        if self.outcome_model is None:
            raise ValueError("Outcome model not loaded. Please load models first.")
        
        # Preprocess patient data
        processed_data = self.preprocess_patient_data(patient_data)
        
        # If recommended treatment is provided, add it to the data
        if recommended_treatment is not None:
            # Encode the recommended treatment
            if 'recommended_treatment' in self.preprocessor.label_encoders:
                le = self.preprocessor.label_encoders['recommended_treatment']
                if recommended_treatment in le.classes_:
                    processed_data['recommended_treatment'] = le.transform([recommended_treatment])[0]
                else:
                    processed_data['recommended_treatment'] = 0  # Default for unseen treatment
            else:
                # Simple mapping if no encoder exists
                treatment_mapping = {treatment: i for i, treatment in enumerate(self.treatment_labels)}
                processed_data['recommended_treatment'] = treatment_mapping.get(recommended_treatment, 0)
        
        # Prepare features (exclude patient_id)
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['patient_id', 'outcome']]
        X = processed_data[feature_columns]
        
        # Scale features
        X_scaled = self.preprocessor.scaler.transform(X)
        
        # Make prediction
        outcome_pred = self.outcome_model.predict(X_scaled)
        
        # Get prediction probabilities if requested
        if return_probabilities and hasattr(self.outcome_model, 'predict_proba'):
            outcome_proba = self.outcome_model.predict_proba(X_scaled)
            return outcome_pred, outcome_proba
        
        return outcome_pred
    
    def get_comprehensive_recommendation(self, patient_data):
        """Get comprehensive treatment recommendation with outcome prediction."""
        results = {}
        
        # Get treatment recommendation with probabilities
        treatment_pred, treatment_proba = self.recommend_treatment(patient_data, return_probabilities=True)
        
        # Convert prediction to treatment name
        recommended_treatment = self.treatment_labels[treatment_pred[0]]
        results['recommended_treatment'] = recommended_treatment
        
        # Get treatment confidence scores
        treatment_confidence = {}
        for i, treatment in enumerate(self.treatment_labels):
            treatment_confidence[treatment] = treatment_proba[0][i]
        results['treatment_confidence'] = treatment_confidence
        
        # Predict outcome for the recommended treatment
        outcome_pred, outcome_proba = self.predict_outcome(
            patient_data, recommended_treatment, return_probabilities=True
        )
        
        # Convert prediction to outcome name
        predicted_outcome = self.outcome_labels[outcome_pred[0]]
        results['predicted_outcome'] = predicted_outcome
        
        # Get outcome confidence scores
        outcome_confidence = {}
        for i, outcome in enumerate(self.outcome_labels):
            outcome_confidence[outcome] = outcome_proba[0][i]
        results['outcome_confidence'] = outcome_confidence
        
        return results
    
    def explain_recommendation(self, patient_data, results):
        """Provide explanation for the recommendation."""
        explanation = []
        
        # Patient profile summary
        explanation.append("Patient Profile Analysis:")
        explanation.append(f"- Age: {patient_data.get('age', 'N/A')}")
        explanation.append(f"- Gender: {patient_data.get('gender', 'N/A')}")
        explanation.append(f"- Diagnosis: {patient_data.get('diagnosis', 'N/A')}")
        explanation.append(f"- Symptoms: {patient_data.get('symptoms', 'N/A')}")
        explanation.append(f"- Severity: {patient_data.get('severity', 'N/A')}")
        explanation.append(f"- Previous Treatment: {patient_data.get('previous_treatment', 'N/A')}")
        
        # Recommendation explanation
        explanation.append(f"\nRecommended Treatment: {results['recommended_treatment']}")
        explanation.append(f"Confidence: {results['treatment_confidence'][results['recommended_treatment']]:.2%}")
        
        # Alternative treatments
        sorted_treatments = sorted(results['treatment_confidence'].items(), 
                                 key=lambda x: x[1], reverse=True)
        explanation.append("\nAlternative Treatments (by confidence):")
        for treatment, confidence in sorted_treatments[1:]:
            explanation.append(f"- {treatment}: {confidence:.2%}")
        
        # Outcome prediction
        explanation.append(f"\nPredicted Outcome: {results['predicted_outcome']}")
        explanation.append(f"Confidence: {results['outcome_confidence'][results['predicted_outcome']]:.2%}")
        
        # Outcome probabilities
        explanation.append("\nOutcome Probabilities:")
        for outcome, probability in results['outcome_confidence'].items():
            explanation.append(f"- {outcome}: {probability:.2%}")
        
        return "\n".join(explanation)


def create_sample_patient():
    """Create a sample patient for testing."""
    return {
        'patient_id': 9999,
        'age': 45,
        'gender': 'Female',
        'symptoms': 'Fatigue',
        'diagnosis': 'Diabetes',
        'previous_treatment': 'Medication A',
        'severity': 'Moderate'
    }


if __name__ == "__main__":
    # Test the recommendation system
    print("Testing Treatment Recommendation System...")
    
    # Create sample patient
    sample_patient = create_sample_patient()
    print(f"Sample Patient: {sample_patient}")
    
    # Initialize recommendation system
    recommender = TreatmentRecommendationSystem()
    
    # Note: In a real scenario, you would load pre-trained models
    print("\nNote: To use this system, you need to:")
    print("1. Train models using train_model.py")
    print("2. Load the trained models using load_models() method")
    print("3. Then use get_comprehensive_recommendation() for new patients")
    
    # Example usage (commented out since models need to be trained first)
    """
    # Load trained models
    success = recommender.load_models(
        'models/treatment_recommendation_model.pkl',
        'models/outcome_prediction_model.pkl'
    )
    
    if success:
        # Get comprehensive recommendation
        results = recommender.get_comprehensive_recommendation(sample_patient)
        
        # Print results
        print("\nRecommendation Results:")
        print(f"Recommended Treatment: {results['recommended_treatment']}")
        print(f"Predicted Outcome: {results['predicted_outcome']}")
        
        # Get detailed explanation
        explanation = recommender.explain_recommendation(sample_patient, results)
        print(f"\nDetailed Explanation:\n{explanation}")
    """