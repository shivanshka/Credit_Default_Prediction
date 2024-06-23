from collections import namedtuple

DataIngestionConfig = namedtuple(
    "DataIngestionConfig",
    [
        "dataset_download_url",
        "zipped_download_dir",
        "raw_data_dir",
        "ingested_train_dir",
        "ingested_test_dir",
    ],
)

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple(
    "DataValidationConfig",
    ["schema_file_path", "report_file_path", "report_page_file_path"],
)

DataTransformationConfig = namedtuple(
    "DataTransformationConfig",
    ["transformed_train_dir", "transformed_test_dir", "preprocessed_object_file_path"],
)

ModelTrainerConfig = namedtuple(
    "ModelTrainerConfig",
    ["trained_model_file_path", "base_auc", "model_config_file_path"],
)

InitializedModelDetail = namedtuple(
    "InitializedModelDetail",
    ["model_serial_number", "model", "param_grid_search", "model_name"],
)

GridSearchBestModel = namedtuple(
    "GridSearchBestModel",
    ["model_serial_number", "model", "best_model", "best_parameters", "best_score"],
)

BestModel = namedtuple(
    "BestModel",
    ["model_serial_number", "model", "best_model", "best_parameters", "best_score"],
)

ModelEvaluationConfig = namedtuple(
    "ModelEvaluationConfig", ["model_evaluation_file_path", "time_stamp"]
)

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])

Experiment = namedtuple(
    "Experiment",
    [
        "experiment_id",
        "initialization_timestamp",
        "artifact_time_stamp",
        "running_status",
        "start_time",
        "stop_time",
        "execution_time",
        "message",
        "experiment_file_path",
        "auc_score",
        "is_model_accepted",
    ],
)
