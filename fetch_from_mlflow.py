import os
import sys
import time
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
def with_retries(func, retries=5, base_delay=1, max_delay=10):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            wait = min(max_delay, base_delay * (2 ** i))
            print(f"[Retry {i+1}/{retries}] Error: {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Max retries reached.")

# ----------------------------------------------------------
# FASTEST WAY TO GET LATEST VERSION
# ----------------------------------------------------------
def get_latest_version(client, name):
    def _call():
        model = client.get_registered_model(name)
        versions = model.latest_versions
        if not versions:
            raise MlflowException(f"No versions found for {name}")

        # sort in descending version number
        return max(versions, key=lambda v: int(v.version))

    print(f"Fetching latest model version for: {name}")
    return with_retries(_call)

# ----------------------------------------------------------
# CHECK IF WE ALREADY DOWNLOADED THIS VERSION
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
# SAVE DOWNLOADED VERSION
# ----------------------------------------------------------
def save_version(version):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))

# ----------------------------------------------------------
# SAFE ARTIFACT DOWNLOAD WITH FALLBACK
# ----------------------------------------------------------
def download_artifact_if_needed(client, version_obj):
    version = version_obj.version
    run_id = version_obj.run_id

    if is_already_downloaded(version):
        print(f"✔ Model v{version} already downloaded. Skipping.")
        return os.path.join(LOCAL_DIR, ARTIFACT_SUBPATH)

    print(f"⬇ Downloading model v{version} (Run ID: {run_id})...")

    # Try MLflow Python API first
    def _download_python_api():
        return mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ARTIFACT_SUBPATH,
            dst_path=LOCAL_DIR,
            timeout=30  # protect from long hangs
        )

    try:
        path = with_retries(_download_python_api)
        save_version(version)
        print(f"✔ Downloaded to: {path}")
        return path
    except Exception as py_err:
        print("⚠ Python MLflow API failed. Trying REST fallback...", py_err)

    # ----------------------------------------------------------
    # FALLBACK: DIRECT REST DOWNLOAD (MORE STABLE)
    # ----------------------------------------------------------
    def _download_rest():
        url = f"{URI}/api/2.0/mlflow-artifacts/artifacts/{run_id}/artifacts/{ARTIFACT_SUBPATH}"
        print(f"REST URL → {url}")

        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            raise RuntimeError(f"REST request failed: {r.status_code} - {r.text}")

        # Write artifact
        os.makedirs(LOCAL_DIR, exist_ok=True)
        out_path = os.path.join(LOCAL_DIR, ARTIFACT_SUBPATH)

        with open(out_path, "wb") as f:
            f.write(r.content)

        return out_path

    path = with_retries(_download_rest)
    save_version(version)
    print(f"✔ REST fallback downloaded to: {path}")
    return path

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    mlflow.set_tracking_uri(URI)
    client = MlflowClient()
    print(f"MLflow tracking URI set to: {URI}")

    try:
        version_obj = get_latest_version(client, NAME)
        print(f"Latest → v{version_obj.version}, Run: {version_obj.run_id}")
    except Exception as e:
        print(f"❌ Failed to get version: {e}")
        sys.exit(1)

    try:
        download_artifact_if_needed(client, version_obj)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

    print("✔ Fetch completed successfully.")

# ----------------------------------------------------------
# ENTRY
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)
