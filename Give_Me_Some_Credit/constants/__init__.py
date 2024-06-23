import os
import sys
from datetime import datetime

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE)

SCHEMA_FILE = "schema.yaml"
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, SCHEMA_FILE)

MODEL_FILE = "model.yaml"
MODEL_YAML_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, MODEL_FILE)

## Training related variables
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_ARTIFACT_DIR = "artifact_dir"

## Data Ingestion related variables
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DIR_KEY = "raw_data_dir"
DATA_INGESTION_ZIPPED_DIR_KEY = "zipped_download_dir"
DATA_INGESTION_ARTIFACT_DIR_KEY = "data_ingestion"
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTED_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTED_TEST_DIR_KEY = "ingested_test_dir"

## Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME = "report_page_file_name"

## Data Transformation/Feature-Engineering variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_DIR_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_KEY = "transformed_train"
DATA_TRANSFORMATION_TEST_DIR_KEY = "transformed_test"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessed_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_OBJ_KEY = "preprocessed_object_file_name"

## Feature-Engineering Constants
AGE_UPPER_QUANTILE = 0.99
AGE_LOWER_QUANTILE = 0.01
RARE_LABEL_VARS_TOLERANCE = 0.20
CUBEROOT_AGE_POWER = 1 / 3
CUBEROOT_REVOLVING_UTILIZATION_OF_UNSECURED_LINES_POWER = 1 / 3

## Renaming features, Columns to use, features to do feature-engineering on
VARIABLES_TO_RENAME_KEY = "variables_to_rename"
USE_COLS_KEY = "use_cols"
OUTLIER_REMOVAL_VARS_KEY = "outlier_removal_vars"
RARE_LABEL_VARS_WITH_SAME_TOLERANCE_KEY = "rare_label_vars_with_same_tolerance"
CUBE_ROOT_OF_AGE_KEY = "cube_root_of_age"
CUBE_ROOT_OF_REVOLVING_UTILIZATION_OF_UNSECURED_LINES_KEY = (
    "cube_root_of_revolving_utilization_of_unsecured_lines"
)
TOTAL_NUMBER_OF_DUES_KEY = "total_number_of_dues"

## Model Trainer related Varibales
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_ARTIFACT_DIR_KEY = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_BASE_AUC_KEY = "base_auc"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_CONFIG_FILE_NAME_KEY = "model_config_file_name"

## Grid-Search related variables
GRID_SEARCH_KEY = "grid_search"
MODULE_KEY = "module"
CLASS_KEY = "class"
PARAMS_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
SEARCH_PARAM_KEY_GRID = "search_param_grid"

## Model Evaluation related variables
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"

## Model Pusher related variables
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

EXPERIMENT_DIR_NAME = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"
TARGET_COLUMN_NAME = "target_column"

THRESHOLD = 0.08928615463036814
