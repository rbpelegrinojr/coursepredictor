"""
Script to create a placeholder logistic regression model for the Course Predictor.
Run this script once to generate `log_model.pkl` if it does not already exist.
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'log_model.pkl')


def create_placeholder_model():
    """Create and save a placeholder logistic regression model."""
    np.random.seed(42)
    n_samples = 500

    # Feature columns matching app.py
    feature_cols = ['LANGUAGE', 'SCIENCE', 'GENERAL_KNOWLEDGE', 'MATH', 'TOTAL',
                    'SHS_GENERAL_AVERAGE', 'SHS_STRAND', 'TIER']

    # Course labels (1-9)
    n_courses = 9

    X = np.column_stack([
        np.random.uniform(10, 40, n_samples),   # LANGUAGE
        np.random.uniform(10, 40, n_samples),   # SCIENCE
        np.random.uniform(10, 40, n_samples),   # GENERAL_KNOWLEDGE
        np.random.uniform(10, 40, n_samples),   # MATH
        np.random.uniform(40, 160, n_samples),  # TOTAL
        np.random.uniform(75, 100, n_samples),  # SHS_GENERAL_AVERAGE
        np.random.randint(1, 16, n_samples),    # SHS_STRAND
        np.random.randint(1, 4, n_samples),     # TIER
    ])

    y = np.random.randint(1, n_courses + 1, n_samples)

    df = pd.DataFrame(X, columns=feature_cols)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42,
                                   solver='lbfgs'))
    ])
    pipeline.fit(df, y)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Placeholder model saved to {MODEL_PATH}")


if __name__ == '__main__':
    create_placeholder_model()
