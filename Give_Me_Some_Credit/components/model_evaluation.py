import os
import sys
import pandas as pd
from Give_Me_Some_Credit.constants import *
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.util.util import read_yaml_file, load_object, write_yaml_file
from Give_Me_Some_Credit.entity.config_entity import ModelEvaluationConfig
from Give_Me_Some_Credit.entity.model_factory import evaluate_classification_model
from Give_Me_Some_Credit.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            logging.info(f"{'>>'*10} Model Evaluation log has started {'<<'*10}.")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = (
                self.model_evaluation_config.model_evaluation_file_path
            )

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)
                return model
            model_eval_file_content = read_yaml_file(
                file_path=model_evaluation_file_path
            )
            model_eval_file_content = (
                dict() if model_eval_file_content is None else model_eval_file_content
            )
            if BEST_MODEL_KEY in model_eval_file_content:
                return model
            model = load_object(
                file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY]
            )
            return model
        except Exception as e:
            raise CreditException(e, sys) from e

    def update_evaluation_report(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = (
                dict() if model_eval_content is None else model_eval_content
            )

            previous_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_model = model_eval_content[BEST_MODEL_KEY]
            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }
            if previous_model is not None:
                model_history = {
                    self.model_evaluation_config.time_stamp: previous_model
                }
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result: {model_eval_content}.")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = (
                self.model_trainer_artifact.trained_model_file_path
            )
            trained_model_object = load_object(trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_NAME]

            logging.info(
                "Separating X and Y features from Training and Testing datasets."
            )
            y_train = train_df[target_column_name]
            y_test = test_df[target_column_name]
            x_train = train_df.drop(columns=[target_column_name], axis=1)
            x_test = test_df.drop(columns=[target_column_name], axis=1)

            model = self.get_best_model()
            if model is None:
                logging.info(
                    "Not found any existing model. Hence, accepting trained model."
                )
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True, evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(
                    f"Model accepted, Model evaluation artifact: {model_evaluation_artifact} created."
                )
                return model_evaluation_artifact
            model_list = [model, trained_model_object]
            metric_info_artifact = evaluate_classification_model(
                model_list=model_list,
                X_test=x_test,
                X_train=x_train,
                y_train=y_train,
                y_test=y_test,
                base_auc=0.8,
            )
            logging.info(
                f"Model evaluation completed, model metric artifact: {metric_info_artifact}."
            )
            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path,
                )
                logging.info(response)
                return response
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True, evaluated_model_path=train_file_path
                )
                self.update_evaluation_report(
                    model_evaluation_artifact=model_evaluation_artifact
                )
                logging.info(
                    f"Model accepted and model evaluation artifact: {model_evaluation_artifact}."
                )
            else:
                logging.info(
                    f"Trined model is not better than the existing model, hence not accepting trained model."
                )
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path,
                )
            return model_evaluation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f'{">>"*10} Model Evalation Config log completed {"<<"*10}.')
