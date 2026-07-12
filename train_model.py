import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

def load_data(filepath):
    """Loads dataset from the specified path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_and_split(df):
    """Preprocesses data and splits into train and test sets."""
    # Split features and target
    X = df.drop(columns=['Approved'])
    y = df['Approved']
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    
    # Train-test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Preprocessing pipelines for numerical and categorical data
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_model(y_true, y_pred, model_name):
    """Computes and prints model evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n==================== {model_name} ====================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    # Paths
    dataset_path = 'd:/skillwallet/CreditCardApproval/dataset/credit_card_data.csv'
    model_output_path = 'd:/skillwallet/CreditCardApproval/model.pkl'
    
    # Load dataset
    print("Loading dataset...")
    df = load_data(dataset_path)
    
    # Print dataset details
    print(f"Dataset Shape: {df.shape}")
    
    # Separate and split
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_f1 = 0
    best_model_name = None
    best_pipeline = None
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline including preprocessing and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, name)
        results[name] = metrics
        
        # Track best model based on F1-score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with F1-score of {best_f1:.4f}")
    
    # Save the entire best pipeline (preprocessor + model)
    print(f"Saving best pipeline to {model_output_path}...")
    with open(model_output_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print("Model saved successfully!")
    
if __name__ == '__main__':
    # Fix import typo in random state argument for train_test_split
    # Let's write the code correctly in python
    import random
    np.random.seed(42)
    main()
