import os
import sys
import time
import json
import mlflow
import requests
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ---------------------------
# CONFIG
# ---------------------------
URI = "http://35.202.220.108:5000/"
NAME = "iris-random-forest"
ARTIFACT_SUBPATH = "random_forest_model"
LOCAL_DIR = "./downloaded_models"
VERSION_FILE = os.path.join(LOCAL_DIR, "latest_version.txt")

# ---------------------------
# SIMPLE RETRY WRAPPER
# ---------------------------
def with_retries(func, retries=5, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"[Retry {i+1}/{retries}] Error: {e}")
            time.sleep(delay)
    raise RuntimeError("Max retries reached for MLflow API call.")

# ---------------------------
# FETCH LATEST MODEL VERSION FAST
# ---------------------------
def get_latest_version(client, name):
    def _call():
        # Use faster low-level API instead of heavy search
        models = client.search_registered_models(filter_string=f"name='{name}'")
        if not models:
            raise MlflowException(f"No registered model named '{name}' found.")

        versions = models[0].latest_versions
        if not versions:
            raise MlflowException(f"No versions available for model '{name}'.")

        # Get highest version
        versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
        return versions_sorted[0]

    print(f"Fetching latest model version for: {name}")
    return with_retries(_call)

# ---------------------------
# CHECK IF ALREADY DOWNLOADED
# ---------------------------
def is_already_downloaded(version_number):
    if not os.path.exists(VERSION_FILE):
        return False

    try:
        with open(VERSION_FILE, "r") as f:
            saved_version = f.read().strip()
        return saved_version == str(version_number)
    except:
        return False

# ---------------------------
# RECORD DOWNLOADED VERSION
# ---------------------------
def save_version(version_number):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(str(version_number))

# ---------------------------
# DOWNLOAD ARTIFACT ONLY IF NEEDED
# ---------------------------
def download_artifact_if_needed(client, version):
    version_number = version.version
    run_id = version.run_id

    if is_already_downloaded(version_number):
        print(f"✔ Model v{version_number} already downloaded. Skipping download.")
        return os.path.join(LOCAL_DIR, ARTIFACT_SUBPATH)

    print(f"⬇ Downloading model v{version_number} (Run ID: {run_id})...")

    def _download():
        return mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ARTIFACT_SUBPATH,
            dst_path=LOCAL_DIR
        )

    downloaded_path = with_retries(_download)
    save_version(version_number)
    
    print(f"✔ Downloaded model to: {downloaded_path}")
    return downloaded_path

# ---------------------------
# MAIN WORKFLOW
# ---------------------------
def main():
    mlflow.set_tracking_uri(URI)
    print(f"MLflow tracking URI set to: {URI}")

    client = MlflowClient()

    try:
        version = get_latest_version(client, NAME)
        print(f"Latest version: v{version.version}, Run ID: {version.run_id}")
    except Exception as e:
        print(f"Failed to get version: {e}")
        sys.exit(1)

    try:
        download_artifact_if_needed(client, version)
    except Exception as e:
        print(f"Artifact download failed: {e}")
        sys.exit(1)

    print("✔ Fetch operation completed successfully.")

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}")
        sys.exit(1)
