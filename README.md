# Week 8: Graded Assignment 8

### Assignment Objectives

1. Integrate data poisoning for IRIS using randomly generated numbers at various levels(5%,10%,50%) and explain the validation outcomes when trained on such data using MLFlow

2. Give your thoughts on how to mitigate such a poisoning attacks and how data quantity requirements evolve when data quality is affected


### Assumptions and Setting

- ✅ Loads `data.csv`

- ✅ Accepts `--label_noise`

- ✅ Applies label + feature noise (set as standard)

- ✅ Trains clean model vs noisy model

- ✅ Evaluates both


### What `fetch_from_mlflow.py` does

- Sets MLflow Tracking URI

- Creates an MlflowClient instance

- Searches for the latest version of the model

- Downloads the model artifacts

    - It will create a folder:

        `downloaded_models/random_forest_model/`

        inside your local machine.

    - This contains:

        - MLmodel file

        - model.pkl (RandomForestClassifier)

        - conda.yaml

        - requirements.txt

        - metadata