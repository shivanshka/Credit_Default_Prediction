import os
import sys
import pip
import json
from flask import request, Flask
from matplotlib.style import context
from Give_Me_Some_Credit.constants import *
from flask import send_file, render_template, abort
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.logger import logging, get_log_dataframe
from Give_Me_Some_Credit.config.configuration import Configuration
from Give_Me_Some_Credit.util.util import read_yaml_file, write_yaml_file
from Give_Me_Some_Credit.pipeline.training_pipeline import Pipeline
from Give_Me_Some_Credit.pipeline.prediction_pipeline import (
    DefaultData,
    DefaultPredictor,
)

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "Give_Me_Some_Credit"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

DEFAULT_DATA_KEY = "credit_data"
SERIOUSDLQIN2YRS_KEY = "SeriousDlqin2yrs"

app = Flask(__name__)


@app.route("/artifact", defaults={"req_path": "Give_Me_Some_Credit"})
@app.route("/artifact/<path:req_path>")
def render_artifact_dir(req_path):
    os.makedirs("housing", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ""
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {
        os.path.join(abs_path, file_name): file_name
        for file_name in os.listdir(abs_path)
        if "artifact" in os.path.join(abs_path, file_name)
    }

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("files.html", result=result)


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)


@app.route("/view_experiment_hist", methods=["GET", "POST"])
def view_experiment_history():
    experiment_df = Pipeline.get_experiment_status()
    context = {
        "experiment": experiment_df.to_html(classes="table table-striped col-12")
    }
    return render_template("experiment_history.html", context=context)


@app.route("/train", methods=["GET", "POST"])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(timestamp=CURRENT_TIME_STAMP))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiment_status().to_html(
            classes="table table-striped col-12"
        ),
        "message": message,
    }
    return render_template("train.html", context=context)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = {DEFAULT_DATA_KEY: None, SERIOUSDLQIN2YRS_KEY: None}

    if request.method == "POST":
        RevolvingUtilizationOfUnsecuredLines = float(
            request.form["RevolvingUtilizationOfUnsecuredLines"]
        )
        age = int(request.form["age"])
        NumberOfTime30_59DaysPastDueNotWorse = int(
            request.form["NumberOfTime30_59DaysPastDueNotWorse"]
        )
        DebtRatio = float(request.form["DebtRatio"])
        MonthlyIncome = float(request.form["MonthlyIncome"])
        NumberOfOpenCreditLinesAndLoans = int(
            request.form["NumberOfOpenCreditLinesAndLoans"]
        )
        NumberOfTimes90DaysLate = int(request.form["NumberOfTimes90DaysLate"])
        NumberRealEstateLoansOrLines = int(request.form["NumberRealEstateLoansOrLines"])
        NumberOfTime60_89DaysPastDueNotWorse = int(
            request.form["NumberOfTime60_89DaysPastDueNotWorse"]
        )
        NumberOfDependents = float(request.form["NumberOfDependents"])

        default_data = DefaultData(
            RevolvingUtilizationOfUnsecuredLines=RevolvingUtilizationOfUnsecuredLines,
            age=age,
            NumberOfTime30_59DaysPastDueNotWorse=NumberOfTime30_59DaysPastDueNotWorse,
            DebtRatio=DebtRatio,
            MonthlyIncome=MonthlyIncome,
            NumberOfOpenCreditLinesAndLoans=NumberOfOpenCreditLinesAndLoans,
            NumberOfTimes90DaysLate=NumberOfTimes90DaysLate,
            NumberRealEstateLoansOrLines=NumberRealEstateLoansOrLines,
            NumberOfTime60_89DaysPastDueNotWorse=NumberOfTime60_89DaysPastDueNotWorse,
            NumberOfDependents=NumberOfDependents,
        )
        credit_df = default_data.get_credit_input_dataframe()

        default_predictor = DefaultPredictor(model_dir=MODEL_DIR)
        print(default_predictor)
        defaulter = default_predictor.predict(X=credit_df)
        context = {
            DEFAULT_DATA_KEY: default_data.get_credit_data_as_dict(),
            SERIOUSDLQIN2YRS_KEY: defaulter,
        }
        print(context)
        return render_template("predict.html", context=context)
    return render_template("predict.html", context=context)


@app.route("/saved_models", defaults={"req_path": "saved_models"})
@app.route("/saved_models/<path:req_path>")
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("saved_models_files.html", result=result)


@app.route("/update_model_config", methods=["GET", "POST"])
def update_model_config():
    try:
        if request.method == "POST":
            model_config = request.form["new_model_config"]
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template(
            "update_model.html", result={"model_config": model_config}
        )

    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f"/logs", defaults={"req_path": f"{LOG_FOLDER_NAME}"})
@app.route(f"/{LOG_FOLDER_NAME}/<path:req_path>")
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template("log.html", context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("log_files.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
