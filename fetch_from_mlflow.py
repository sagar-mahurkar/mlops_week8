import os
import sys
import time
import mlflow
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
# SIMPLE RETRY
# ----------------------------------------------------------
def with_retries(func, retries=5, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            print(f"[Retry {i+1}/{retries}] Error: {e}")
            time.sleep(delay)
    raise RuntimeError("Max retries reached.")

# ----------------------------------------------------------
# FASTEST WAY TO GET LATEST VERSION
# ----------------------------------------------------------
def get_latest_version(client, name):
    def _call():
        model = client.get_registered_model(name)
        versions = model.latest_versions
        if not versions:
            raise MlflowException(f"No versions found for model '{name}'")
        return max(versions, key=lambda v: int(v.version))
    return with_retries(_call)

# ----------------------------------------------------------
# CHECK IF ALREADY DOWNLOADED
# ----------------------------------------------------------
def is_already_downloaded(version):
    if not os.path.exists(VERSION_FILE):
        return False
    try:
        with open(VERSION_FILE, "r") as f:
            return f.read().strip() == str(version)
    except:
        return False

# ----------------------------------------------------------
# SAVE VERSION
# ----------------------------------------------------------
def save_version(version):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))

# ----------------------------------------------------------
# DOWNLOAD ARTIFACT (ONLY PYTHON API, NO REST)
# ----------------------------------------------------------
def download_artifact_if_needed(version_obj):
    version = version_obj.version
    run_id = version_obj.run_id

    if is_already_downloaded(version):
        print(f"✔ Model v{version} already downloaded. Skipping.")
        return os.path.join(LOCAL_DIR, ARTIFACT_SUBPATH)

    print(f"⬇ Downloading model v{version} (Run ID: {run_id})...")

    def _download():
        return mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ARTIFACT_SUBPATH,
            dst_path=LOCAL_DIR
        )

    downloaded_path = with_retries(_download)
    save_version(version)

    print(f"✔ Downloaded to: {downloaded_path}")
    return downloaded_path

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    mlflow.set_tracking_uri(URI)
    client = MlflowClient()

    try:
        version_obj = get_latest_version(client, NAME)
        print(f"Latest → v{version_obj.version}, Run: {version_obj.run_id}")
    except Exception as e:
        print(f"❌ Failed to get version: {e}")
        sys.exit(1)

    try:
        download_artifact_if_needed(version_obj)
    except Exception as e:
        print(f"❌ Artifact download failed: {e}")
        sys.exit(1)

    print("✔ Fetch completed successfully.")

# ----------------------------------------------------------
# ENTRY
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
