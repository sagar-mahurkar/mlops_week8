import unittest
import pandas as pd
import os


class TestData(unittest.TestCase):

    data_path = './data.csv'

    def setUp(self):
        """Load dataset only"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError("data.csv not found.")
            self.input_data = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Setup failed: {e}")
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

        features_present = all(
            col in self.input_data.columns for col in required_features
        )

        self.assertTrue(features_present, "Required features missing from data")


if __name__ == '__main__':
    unittest.main()
