import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow

# ---------------- Configuration ----------------
MLFLOW_TRACKING_URI = "http://136.112.255.152:5000"
MODEL_NAME = "iris-random-forest"

LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ---------------- Noise Functions ----------------
def add_label_noise(y, noise_frac, random_state=None):
    """Randomly flip noise_frac fraction of labels."""
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()

    if noise_frac <= 0:
        return y_noisy

    n = len(y)
    k = int(np.floor(noise_frac * n))
    idx = rng.choice(n, size=k, replace=False)

    classes = np.unique(y)
    for i in idx:
        curr = y_noisy[i]
        other_choices = classes[classes != curr]
        y_noisy[i] = rng.choice(other_choices)

    return y_noisy


def add_feature_noise(X, noise_std=0.05, random_state=None):
    """Add Gaussian noise to the feature matrix."""
    rng = np.random.RandomState(random_state)
    X_noisy = X.astype(float).copy()
    X_noisy += rng.normal(scale=noise_std, size=X.shape)
    return X_noisy


# ---------------- Data Prep ----------------
def prepare_data(label_noise_frac=0.0):
    print("Loading dataset from ./data.csv ...")
    df = pd.read_csv("./data.csv")
    print("Dataset loaded successfully!")
    print(f"Total rows: {len(df)}")

    print("Splitting data into train/test ...")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values

    X_train, X_test, y_train_clean, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("Data split complete.")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Apply noise
    print(f"Applying label noise: {label_noise_frac}")
    y_train_noisy = add_label_noise(y_train_clean, label_noise_frac, random_state=123)

    print("Applying feature noise...")
    X_train_noisy = add_feature_noise(X_train, noise_std=0.05, random_state=123)

    return X_train_noisy, y_train_noisy, X_test, y_test


# ---------------- Training ----------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    print("Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Tracking URI set to: {MLFLOW_TRACKING_URI}")

    print("Starting Random Forest hyperparameter tuning...")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10],
        "min_samples_split": [3, 5, 10],
        "class_weight": ["balanced", None]
    }

    with mlflow.start_run(run_name="Random Forest Hyperparameter Search"):
        print("Running GridSearchCV...")
        model = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Grid search complete. Best parameters:")
        print(grid.best_params_)

        best = grid.best_estimator_

        print("Logging parameters & metrics to MLflow...")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_accuracy", grid.best_score_)
        mlflow.log_metric("test_accuracy", grid.score(X_test, y_test))

        print("Logging model to MLflow registry...")
        mlflow.sklearn.log_model(best, "model", registered_model_name=MODEL_NAME)

        print(f"Saving model locally at {LOCAL_MODEL_PATH}")
        joblib.dump(best, LOCAL_MODEL_PATH)

        return {
            "best_params": grid.best_params_,
            "cv_accuracy": grid.best_score_,
            "test_accuracy": grid.score(X_test, y_test)
        }


# ---------------- Main ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_noise",
        type=float,
        default=0.0,
        help="Fraction of labels to flip (0.0 to 1.0)"
    )
    args = parser.parse_args()

    print(f"Preparing data with label noise = {args.label_noise}")
    X_train, y_train, X_test, y_test = prepare_data(label_noise_frac=args.label_noise)

    print("\nStarting training...")
    result = tune_random_forest(X_train, y_train, X_test, y_test)

    print("\nTraining complete!")
    print("Results:")
    print(result)
