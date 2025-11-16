import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def main(label_noise_frac):
    # Load data
    df = pd.read_csv("data.csv")

    # Assume last column is label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train-test split (clean test)
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Apply noise
    y_train_noisy = add_label_noise(y_train_clean, label_noise_frac, random_state=123)
    X_train_noisy = add_feature_noise(X_train, noise_std=0.05, random_state=123)

    # Train ONLY noisy model
    clf_noisy = RandomForestClassifier(n_estimators=200, random_state=0)
    clf_noisy.fit(X_train_noisy, y_train_noisy)

    # Evaluate noisy model on clean test
    y_pred_noisy = clf_noisy.predict(X_test)

    print("\n=== Model trained on noisy data ===")
    print("Label noise fraction:", label_noise_frac)
    print("Accuracy:", accuracy_score(y_test, y_pred_noisy))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_noisy))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_noisy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_noise",
        type=float,
        default=0.0,
        help="Fraction of labels to flip (0.0 to 1.0)"
    )

    args = parser.parse_args()
    main(args.label_noise)
