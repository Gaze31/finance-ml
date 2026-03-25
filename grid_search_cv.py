import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
svm_model = SVC()

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available processors
    verbose=1,  # Show progress
    return_train_score=True
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Display results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
print("Test set score:", grid_search.score(X_test_scaled, y_test))

# Random Forest example
rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best CV score:", grid_search_rf.best_score_)

# Get all results as DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Display important columns
important_columns = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(results_df[important_columns].head(10))

# Find the best combination
print("\nBest parameters combination:")
print(results_df[results_df['rank_test_score'] == 1][important_columns])

# Visualize results (for 2 parameters)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_search_results(grid_search, param1, param2):
    """Plot heatmap of GridSearchCV results"""
    results = pd.DataFrame(grid_search.cv_results_)
    scores = results.pivot_table(
        index=f'param_{param1}',
        columns=f'param_{param2}',
        values='mean_test_score'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores, annot=True, cmap='viridis', fmt='.3f')
    plt.title(f'Grid Search Scores: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()

# Example usage
plot_grid_search_results(grid_search, 'C', 'gamma')

from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Define multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

grid_search_multi = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1',  # Refit using best F1 score
    n_jobs=-1,
    return_train_score=True
)

grid_search_multi.fit(X_train_scaled, y_train)

# Access different scores
print("Best parameters:", grid_search_multi.best_params_)
print("Best F1 score:", grid_search_multi.best_score_)
print("Test set accuracy:", grid_search_multi.score(X_test_scaled, y_test))

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())
])

# Define parameter grid for pipeline
param_grid_pipeline = {
    'pca__n_components': [5, 10, 15, 20],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.001, 0.01, 0.1],
    'svm__kernel': ['rbf', 'linear']
}

grid_search_pipeline = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid_pipeline,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_pipeline.fit(X_train, y_train)

print("Best pipeline parameters:", grid_search_pipeline.best_params_)
print("Best CV score:", grid_search_pipeline.best_score_)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Use RandomizedSearchCV for large parameter spaces
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(range(10, 50, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best parameters from Random Search:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

import time

def time_grid_search(grid_search, X, y):
    """Time the grid search execution"""
    start_time = time.time()
    grid_search.fit(X, y)
    elapsed_time = time.time() - start_time
    print(f"Grid search took {elapsed_time:.2f} seconds")
    return elapsed_time

# Use with smaller grid for quicker results
quick_grid = {
    'C': [0.1, 1, 10],  # Reduced from 4 to 3
    'gamma': [0.01, 0.1],  # Reduced from 4 to 2
    'kernel': ['rbf']  # Reduced options
}

grid_search_quick = GridSearchCV(
    estimator=SVC(),
    param_grid=quick_grid,
    cv=3,  # Reduced folds
    n_jobs=-1
)

time_grid_search(grid_search_quick, X_train_scaled, y_train)

def evaluate_best_model(grid_search, X_test, y_test, feature_names=None):
    """Evaluate the best model from grid search"""
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("Best Parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance if available
    if hasattr(best_model, 'feature_importances_') and feature_names is not None:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print("\nTop 10 Feature Importances:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Use the function
evaluate_best_model(grid_search_rf, X_test, y_test, feature_names=data.feature_names)