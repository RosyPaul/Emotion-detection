import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "RosyPaul"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/RosyPaul/mlops-prj1.mlflow")

        cls.new_model_name = "spam-classifier"
        try:
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        except RuntimeError as e:
            raise unittest.SkipTest(str(e))

        # Load model via runs:/ URI to avoid the registry download endpoint
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(
            name=cls.new_model_name,
            version=cls.new_model_version
        )
        run_id = model_version_details.run_id
        cls.new_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # No vectorizer needed — features already in the CSV
        cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")

    @staticmethod
    def get_latest_model_version(model_name: str):
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            return latest.version
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Model '{model_name}' not found in registry. "
                f"Run the training pipeline first. Original error: {e}"
            )

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Use one row from the test data directly — no vectorizer needed
        X_sample = self.holdout_data.iloc[:1, :-1]
        prediction = self.new_model.predict(X_sample)

        # Input shape matches the data
        self.assertEqual(X_sample.shape[0], 1)
        self.assertGreater(X_sample.shape[1], 0)

        # Output is a single prediction for the one input row
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.new_model.predict(X_holdout)

        self.assertGreaterEqual(accuracy_score(y_holdout, y_pred),  0.40, "Accuracy below threshold")
        self.assertGreaterEqual(precision_score(y_holdout, y_pred), 0.40, "Precision below threshold")
        self.assertGreaterEqual(recall_score(y_holdout, y_pred),    0.40, "Recall below threshold")
        self.assertGreaterEqual(f1_score(y_holdout, y_pred),        0.40, "F1 below threshold")

if __name__ == "__main__":
    unittest.main()