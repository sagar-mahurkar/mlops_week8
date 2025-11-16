import unittest
import mlflow
import pandas as pd
from sklearn.metrics import recall_score
import os


class TestModel(unittest.TestCase):
    model = None
    input_data = None

    # Updated paths
    data_path = './data.csv'
    target_column = 'species'

    def setUp(self):
        """Load latest MLflow model + dataset"""
        try:
            # Locate latest MLflow run directory
            mlruns_root = "./mlruns"
            if not os.path.exists(mlruns_root):
                raise FileNotFoundError("mlruns directory not found. Model not trained yet.")

            latest_run_path = None
            latest_ts = -1

            # Iterate through MLflow experiment folders
            for root, dirs, files in os.walk(mlruns_root):
                if "model.pkl" in files:
                    model_path = os.path.join(root, "model.pkl")
                    ts = os.path.getmtime(model_path)
                    if ts > latest_ts:
                        latest_ts = ts
                        latest_run_path = model_path

            if not latest_run_path:
                raise FileNotFoundError("No model.pkl found inside mlruns/")

            print(f"Loading model from: {latest_run_path}")
            self.model = mlflow.pyfunc.load_model(os.path.dirname(latest_run_path))

            # Load dataset
            self.input_data = pd.read_csv(self.data_path)

        except Exception as e:
            print(f"Setup failed: {e}")
            self.model = None
            self.input_data = None

    def test_data_integrity_check(self):
        """Ensure required columns exist in the dataset"""
        self.assertIsNotNone(self.input_data, "Data not loaded properly")

        required_features = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        features_present = all(col in self.input_data.columns for col in required_features)
        self.assertTrue(features_present, "Required features missing from data")

    def test_model_accuracy(self):
        """Model should achieve a minimum recall on Iris dataset"""
        self.assertIsNotNone(self.model, "MLflow model not loaded properly")
        self.assertIsNotNone(self.input_data, "Data not loaded properly")

        df_features = self.input_data.drop(self.target_column, axis=1, errors='ignore')
        target_labels = self.input_data[self.target_column]

        predictions = self.model.predict(df_features)
        retrieval_rate = recall_score(target_labels, predictions, average='macro')

        self.assertGreater(
            retrieval_rate,
            0.85,
            f"Model recall too low: {retrieval_rate}"
        )


if __name__ == '__main__':
    unittest.main()
