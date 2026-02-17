# TruthLens AI – Fake News Classification using NLP & Deep Learning

TruthLens AI is a Natural Language Processing (NLP) project that classifies news articles as Real or Fake using both classical machine learning and deep learning techniques.

This project compares TF-IDF + Logistic Regression with LSTM-based Deep Learning, highlighting differences in performance and modeling approaches.

🚀 Project Objectives

- Build an end-to-end fake news classification pipeline
- Apply both traditional ML and deep learning methods
- Compare model performance and analyze results
- Understand the impact of text preprocessing and feature engineering

📂 Dataset

The project uses the Fake and Real News Dataset from Kaggle:

- Contains labeled news articles (Real = 0, Fake = 1)
- Two CSV files merged and shuffled before training
- Text cleaned and preprocessed before modeling

🔄 Workflow
Data Loading
→ Label Assignment
→ Data Merging & Shuffling
→ Text Cleaning
→ Feature Engineering
→ Model Training
→ Evaluation
→ Model Comparison
→ Conclusion

🧠 Models Implemented
1️⃣ Logistic Regression (TF-IDF)

- Text vectorized using TF-IDF
- Logistic Regression classifier
- Strong baseline model
- Fast and efficient

2️⃣ LSTM (Deep Learning)

- One-hot encoding
- Sequence padding
- Embedding layer
- LSTM layer
- Sigmoid output for binary classification

📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve & AUC
- Model comparison visualization

📈 Key Insights

- TF-IDF + Logistic Regression provides a strong and reliable baseline.
- LSTM captures contextual and sequential information in text.
- Classical ML models remain highly competitive in structured NLP tasks.
- Deep learning models become more beneficial with larger datasets and complex patterns.

🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

📌 Project Structure
TruthLens-AI/
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── truthlens_fake_news_classifier.ipynb
└── README.md

🎯 Final Outcome

- This project demonstrates a complete NLP classification pipeline, from preprocessing to deep learning, with a comparative analysis of classical and neural approaches.

Author:
Pulkit Chhabra
