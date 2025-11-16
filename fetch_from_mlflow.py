import os
import sys
import time
import random
import mlflow
import requests
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
URI = "http://104.154.76.71:5000/"
NAME = "iris-random-forest"
ARTIFACT_SUBPATH = "random_forest_model"
LOCAL_DIR = "./downloaded_models"
VERSION_FILE = os.path.join(LOCAL_DIR, "latest_version.txt")

# ----------------------------------------------------------
# EXPONENTIAL BACKOFF
# ----------------------------------------------------------
def retry(func, retries=6, base_delay=1, max_delay=8):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            # Retry only on network/internal errors
            if not should_retry(e):
                raise

            sleep = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.3)
            print(f"[Retry {attempt+1}/{retries}] Error: {e} → sleeping {sleep:.1f}s")
            time.sleep(sleep)

    raise RuntimeError("Max retries reached")


def should_retry(err):
    """Retry only if error is network-related or MLflow server 5xx."""
    if isinstance(err, requests.exceptions.RequestException):
        return True

    msg = str(err).lower()
    retriable_keywords = ["500", "timeout", "temporarily", "connection", "broken pipe"]

    return any(k in msg for k in retriable_keywords)

# ----------------------------------------------------------
# GET LATEST VERSION SAFELY
# ----------------------------------------------------------
def get_latest_version(client, name):

    def _call():
        model = client.get_registered_model(name)

        # Fallback: Sometimes latest_versions is empty or partial.
        if not model.latest_versions:
            versions = client.search_model_versions(f"name='{name}'")
            versions = list(versions)
            if not versions:
                raise MlflowException(f"No versions found for model '{name}'")
            return max(versions, key=lambda v: int(v.version))

        # Normal path
        return max(model.latest_versions, key=lambda v: int(v.version))

    return retry(_call)

# ----------------------------------------------------------
# CHECK EXISTING VERSION
# ----------------------------------------------------------
def is_already_downloaded(version):
    if not os.path.exists(VERSION_FILE):
        return False
    try:
        with open(VERSION_FILE, "r") as f:
            return f.read().strip() == str(version)
    except:
        return False

def save_version(version):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))

# ----------------------------------------------------------
# DOWNLOAD ARTIFACT
# ----------------------------------------------------------
def download_artifact(version_obj):

    version = version_obj.version
    run_id = version_obj.run_id

    if is_already_downloaded(version):
        print(f"✔ Model v{version} already present. Skipping download.")
        return os.path.join(LOCAL_DIR, ARTIFACT_SUBPATH)

    print(f"⬇ Downloading model v{version} (run_id: {run_id})...")

    def _download():
        # Explicitly use artifact URI
        uri = f"runs:/{run_id}/{ARTIFACT_SUBPATH}"
        return mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=LOCAL_DIR
        )

    downloaded = retry(_download)
    save_version(version)
    print(f"✔ Downloaded to: {downloaded}")

    return downloaded

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    mlflow.set_tracking_uri(URI)
    client = MlflowClient()

    print("Connecting to model registry...")

    try:
        version_obj = get_latest_version(client, NAME)
        print(f"Latest → v{version_obj.version} (run_id={version_obj.run_id})")
    except Exception as e:
        print(f"❌ Failed to get latest version: {e}")
        sys.exit(1)

    try:
        download_artifact(version_obj)
    except Exception as e:
        print(f"❌ Artifact download error: {e}")
        sys.exit(1)

    print("✔ Completed successfully")

# ----------------------------------------------------------
# ENTRY
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
