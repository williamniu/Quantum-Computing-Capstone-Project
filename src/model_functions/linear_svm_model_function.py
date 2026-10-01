import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline


def linear_svm_model_function(train_df, test_df, y_train):
    y = y_train.reindex(train_df.index)
    mask = train_df.notna().all(axis=1) & y.notna()

    X_train = train_df.loc[mask].values
    y_fit   = y.loc[mask].values
    X_test  = test_df.fillna(train_df.median()).values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVR(C=1.0, epsilon=0.1, max_iter=20000, random_state=42)),
    ])
    pipeline.fit(X_train, y_fit)

    return pipeline.predict(X_test), test_df.index
