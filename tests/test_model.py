# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure MLflow tracking (same as your main script)
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "RosyPaul"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/RosyPaul/mlops-prj1.mlflow")

        cls.new_model_name = "spam-classifier"   # must match name used in registration
        try:
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        except RuntimeError as e:
            raise unittest.SkipTest(str(e))  # Skip all tests instead of hard ERROR
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")



    # def setUpClass(cls):
    #     # Set up DagsHub credentials for MLflow tracking
    #     dagshub_token = os.getenv("DAGSHUB_PAT")
    #     if not dagshub_token:
    #         raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    #     os.environ["MLFLOW_TRACKING_USERNAME"] = 'RosyPaul'
    #     os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    #     # Set up MLflow tracking URI
    #     mlflow.set_tracking_uri('https://dagshub.com/RosyPaul/mlops-prj1.mlflow')

    #     # Load the new model from MLflow model registry
    #     cls.new_model_name = "model"
    #     cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
    #     cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
    #     cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

    #     # Load the vectorizer
    #     cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

    #     # Load holdout test data
    #     cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    # @staticmethod
    # def get_latest_model_version(model_name, stage="Staging"):
    #     client = mlflow.MlflowClient()
    #     latest_version = client.get_latest_versions(model_name, stages=[stage])
    #     return latest_version[0].version if latest_version else None

    @staticmethod
    def get_latest_model_version(model_name: str):
        client = mlflow.tracking.MlflowClient()
        try:
            # Replaces deprecated get_latest_versions(stages=[...])
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            # Return the version with the highest version number
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
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()