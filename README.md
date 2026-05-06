# High-Performance-Stroke-Prediction-Using-Catboost-Explainable-ML-Models

# High-Performance Stroke Prediction Using CatBoost and Explainable AI

Developed an advanced healthcare-based machine learning system for early stroke prediction and risk assessment using ensemble learning and Explainable AI techniques. The project focuses on improving prediction accuracy, handling imbalanced medical datasets, and providing transparent model interpretations for better clinical decision-making.

Collected and preprocessed clinical healthcare data by handling missing values, performing label encoding, feature normalization, and exploratory data analysis using Pandas, NumPy, Matplotlib, and Seaborn. Implemented SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance and improve model generalization on unseen data.

Built and compared multiple machine learning models including Naive Bayes, Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, XGBoost, and CatBoost. The CatBoost ensemble model delivered the best performance with an accuracy of 95.38%, outperforming traditional classification techniques in stroke risk prediction.

Integrated Explainable Artificial Intelligence (XAI) using SHAP (SHapley Additive Explanations) to provide both global and local interpretability of predictions. This helped identify important clinical factors influencing stroke risk and improved the transparency and trustworthiness of the system for healthcare applications.

Designed and deployed a secure Flask-based web application that allows users to register, log in, and perform real-time stroke risk prediction through an interactive user interface. Integrated SQLite database functionality for secure authentication and data management. The application enables fast, scalable, and user-friendly prediction suitable for practical healthcare deployment.

### Key Features

* Automated stroke risk prediction system
* Ensemble learning using CatBoost
* Class imbalance handling with SMOTE
* Explainable AI integration using SHAP
* Real-time prediction through Flask web application
* Secure user authentication with SQLite
* Data visualization and feature analysis
* High prediction accuracy of 95.38%

### Technologies Used

Python, Machine Learning, CatBoost, XGBoost, Scikit-learn, SHAP, Flask, SQLite, Pandas, NumPy, Matplotlib, Seaborn, SMOTE
