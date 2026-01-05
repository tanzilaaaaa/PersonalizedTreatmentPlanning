# 🏥 AI Healthcare Assistant

A machine learning system that predicts treatment outcomes using patient data. Built for educational and research purposes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)

## ✨ Features

- **Web Interface** - Modern UI for patient data input and predictions
- **ML Model** - Random Forest classifier with 35.33% accuracy (+5.7% improvement)
- **Real-time Predictions** - Instant treatment outcome predictions
- **Interactive Charts** - Plotly visualizations for results
- **REST API** - JSON endpoints for programmatic access

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if needed)
python quick_model_improvement.py

# Start web app
python web_app.py

# Open browser to http://localhost:5555
```

## 📁 Project Structure

```
ai-healthcare-assistant/
├── 🌐 Web Application
│   ├── web_app.py                    # Main Flask application with modern UI
│   ├── simple_web_app.py             # Lightweight version for testing
│   ├── templates/                    # HTML templates with Bootstrap 5
│   │   ├── base.html                 # Base template with navigation
│   │   ├── index.html                # Patient input form
│   │   ├── results.html              # Prediction results with charts
│   │   ├── demo.html                 # Sample patients showcase
│   │   └── about.html                # Model information page
│   └── static/                       # CSS, JS, and assets
│
├── 🤖 Machine Learning Core
│   ├── src/
│   │   ├── data_preprocessing.py     # Data cleaning and encoding
│   │   ├── enhanced_feature_engineering.py  # Advanced feature creation
│   │   ├── advanced_model_training.py       # Optimized ML pipeline
│   │   ├── train_model.py            # Basic model training
│   │   ├── evaluate_model.py         # Model performance evaluation
│   │   └── recommend.py              # Treatment recommendation logic
│   │
│   ├── quick_model_improvement.py    # Fast model optimization script
│   ├── improve_model_performance.py  # Comprehensive model enhancement
│   └── demo.py                       # Command-line demonstration
│
├── 📊 Data Analysis & Research
│   ├── notebooks/                    # Jupyter notebooks for exploration
│   │   ├── 01_data_loading.ipynb     # Data import and initial analysis
│   │   ├── 02_eda.ipynb              # Exploratory data analysis
│   │   ├── 03_feature_engineering.ipynb  # Feature creation experiments
│   │   ├── 04_model_training.ipynb   # Model training and comparison
│   │   └── 05_evaluation.ipynb       # Performance evaluation and metrics
│   │
│   └── data/                         # Dataset storage
│       ├── raw/                      # Original unprocessed data
│       └── processed/                # Cleaned and prepared datasets
│
└── 🗄️ Trained Models & Assets
    └── models/                       # Serialized ML models and components
        ├── quick_improved_model.pkl  # Optimized Random Forest model
        ├── quick_feature_selector.pkl # Feature selection transformer
        ├── quick_preprocessor.pkl    # Data preprocessing pipeline
        └── quick_selected_features.json # Selected feature names
```

## 🎯 How It Works

The AI analyzes patient data through a 4-step process:
1. **Data Input** - Patient demographics and medical history
2. **Feature Engineering** - Creates 33 features from basic data  
3. **ML Prediction** - Random Forest model processes 12 key factors
4. **Results** - Predicts Improved/Stable/Not Improved with confidence scores

## �️ Tech Stack

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: Bootstrap 5, Plotly.js, AOS animations
- **ML**: Random Forest, Feature Engineering, Cross-validation

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 35.33% |
| Improvement | +5.7% |
| Features | 12 selected |
| Validation | 5-fold CV |

## ⚠️ Disclaimer

**For educational and research purposes only.** Not intended for actual medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.