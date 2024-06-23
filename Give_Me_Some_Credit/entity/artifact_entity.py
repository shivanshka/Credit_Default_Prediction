from collections import namedtuple

DataIngestionArtifact = namedtuple(
    "DataIngestionArtifact",
    ["train_file_path", "test_file_path", "is_ingested", "message"],
)

DataValidationArtifact = namedtuple(
    "DataValidationArtifact",
    [
        "schema_file_path",
        "report_file_path",
        "report_page_file_path",
        "is_validated",
        "message",
    ],
)

DataTransformationArtifact = namedtuple(
    "DataTransformationArtifact",
    [
        "transformed_train_file_path",
        "transformed_test_file_path",
        "preprocessed_object_file_path",
        "is_transformed",
        "message",
    ],
)

ModelTrainerArtifact = namedtuple(
    "ModelTrainerArtifact",
    [
        "trained_model_file_path",
        "train_log_loss",
        "test_log_loss",
        "train_auc_score",
        "test_auc_score",
        "is_trained",
        "message",
        "threshold",
    ],
)

MetricsInfoArtifact = namedtuple(
    "MetricsInfoArtifact",
    [
        "model_name",
        "model_object",
        "train_log_loss",
        "test_log_loss",
        "train_auc_score",
        "test_auc_score",
        "index_number",
        "threshold",
    ],
)

ModelEvaluationArtifact = namedtuple(
    "ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_path"]
)

ModelPusherArtifact = namedtuple(
    "ModelPusherArtifact", ["is_model_pushed", "model_export_file_path"]
)
