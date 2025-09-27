# Credit Card Fraud Detection Using Tree-Based and Ensemble Models

## Project Overview
This repository contains a comprehensive machine learning project for detecting fraudulent credit card transactions. It addresses a real-world challenge where banks lose billions annually to fraud, treating it as a "needle-in-a-haystack" problem with highly imbalanced data (fraud cases ~0.17%).

The project builds on supervised classification (Decision Trees, Random Forests) and incorporates advanced techniques:
- **Unsupervised Anomaly Detection**: Isolation Forest to identify fraud as outliers.
- **Handling Imbalance**: SMOTE for synthetic oversampling.
- **Dimensionality Reduction**: PCA to reduce features and mitigate overfitting.
- **Advanced Ensembles**: XGBoost for gradient boosting and a Voting Classifier for model combination.
- **Evaluation**: Focus on precision-recall AUC, ROC-AUC, and F1-score (aiming for >0.9 F1 and 95%+ recall).

This enhances basic tree models, achieving robust performance on the Kaggle Credit Card Fraud Detection dataset (~284,807 transactions, 30 features).

### Why This Project?
- **Real-World Impact**: Demonstrates how ML can automate fraud detection, saving time and resources.
- **Skill Showcase**: Covers data preparation, feature engineering, model training/tuning, ensemble methods, and deployment.
- **Resume Boost**: "Developed a fraud detection system using ensembles and anomaly detection; handled imbalance with SMOTE, achieving 95%+ recall on a Kaggle dataset."

## Dataset
- Source: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Details: 284,807 rows, 30 features (PCA-transformed V1-V28, Time, Amount), binary target (Class: 0=legit, 1=fraud).
- **Instructions**: Download `creditcard.csv` from the link and place it in the project root or upload to Colab.

## Prerequisites
- Python 3.8+
- Google Colab or Jupyter Notebook environment.
- Install dependencies: `pip install -r requirements.txt`

## Installation
1. Clone the repo: `git clone https://github.com/Khalifa-Bouneb/credit-card-fraud-detection.git`
2. Install requirements: `pip install -r requirements.txt`
3. Download the dataset as mentioned above.

## Usage
### Training and Evaluation
1. Open `credit_card_fraud_detection.ipynb` in Jupyter/Colab.
2. Run cells sequentially to:
   - Prepare data (scaling, splitting).
   - Apply PCA and SMOTE.
   - Train models (Decision Tree, Random Forest, Isolation Forest, XGBoost).
   - Tune hyperparameters and ensemble.
   - Evaluate with metrics and PR curve.
3. The notebook saves trained models (e.g., `ensemble.pkl`) for deployment.

### Deployment
- Run the Gradio

## Project Structure
- `credit_card_fraud_detection.ipynb`: Full workflow notebook.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Ignores large files like datasets and virtualenvs.

## Results
- Achieved ROC-AUC >0.99 and high recall on fraud class.
- Key features: V14, V17 (from Random Forest importance).

## Extensions
- Submit predictions to Kaggle notebooks for comparison.
- Tune Isolation Forest contamination for better anomaly detection.
- Integrate with APIs for real-time deployment.

## License
MIT License - Feel free to use and contribute.

## Contact
- GitHub: [Khalifa-Bouneb](https://github.com/Khalifa-Bouneb)
- For questions: Open an issue.

This project was developed by Khalifa Bouneb as part of advancing ML skills in fraud detection.
