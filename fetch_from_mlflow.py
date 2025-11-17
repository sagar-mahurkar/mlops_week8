import os
import joblib
import mlflow

MLFLOW_TRACKING_URI = "http://136.112.255.152:5000/"
MODEL_NAME = "iris-random-forest"

LOCAL_MODEL_DIR = "downloaded_models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def load_latest_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, LOCAL_MODEL_PATH)

    print(f"Model downloaded to {LOCAL_MODEL_PATH}")
    return model

if __name__ == "__main__":
    load_latest_model()
