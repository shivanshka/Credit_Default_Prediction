import os
import sys
import numpy as np
import pandas as pd
from Give_Me_Some_Credit.constants import *
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.entity.model_factory import (
    ModelFactory,
    evaluate_classification_model,
)
from Give_Me_Some_Credit.util.util import load_object, save_object
from Give_Me_Some_Credit.entity.config_entity import (
    ModelTrainerConfig,
    GridSearchBestModel,
)
from Give_Me_Some_Credit.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    MetricsInfoArtifact,
)
from Give_Me_Some_Credit.util.util import read_yaml_file, write_yaml_file


class CreditEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_features = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict_proba(transformed_features)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info(f"{'>>'*10} Model training log has started {'<<'*10}.")
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading Transformed Train dataset.")
            transformed_train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            transformed_train_df = pd.read_csv(
                filepath_or_buffer=transformed_train_file_path
            )

            logging.info("Loading Transformed Test dataset.")
            transformed_test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )
            transformed_test_df = pd.read_csv(
                filepath_or_buffer=transformed_test_file_path
            )

            logging.info("Separating X & Y features from Training & Testing datasets.")
            transformed_train_y = transformed_train_df.iloc[:, -1]
            transformed_train_x = transformed_train_df.iloc[:, :-1]
            transformed_test_y = transformed_test_df.iloc[:, -1]
            transformed_test_x = transformed_test_df.iloc[:, :-1]

            logging.info("Extracting Model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(
                f"Initailizing Model Factory class using above model config file: {model_config_file_path}"
            )
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_auc = self.model_trainer_config.base_auc
            logging.info(f"Expected Base AUC: {base_auc}")

            logging.info("Initializing operation model selection.")
            best_model = model_factory.get_best_model(
                X=transformed_train_x, y=transformed_train_y, base_auc=base_auc
            )
            logging.info(f"Best model found on training dataset: {best_model}")
            logging.info("Extracting trained model list.")
            grid_searched_best_model_list = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(
                "Evaluating all trained models on training and testing datasets."
            )
            metric_info = evaluate_classification_model(
                model_list=model_list,
                X_train=transformed_train_x,
                y_train=transformed_train_y,
                X_test=transformed_test_x,
                y_test=transformed_test_y,
                base_auc=base_auc,
            )
            logging.info("Best model found on both training and testing datasets.")

            threshold = metric_info.threshold
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.preprocessed_object_file_path
            )
            model_object = metric_info.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            trained_credit_model = CreditEstimatorModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=model_object,
            )
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=trained_credit_model)

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained Successfully.",
                trained_model_file_path=trained_model_file_path,
                train_log_loss=metric_info.train_log_loss,
                test_log_loss=metric_info.test_log_loss,
                train_auc_score=metric_info.train_auc_score,
                test_auc_score=metric_info.test_auc_score,
                threshold=threshold,
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20} Model Trainer Stage Completed {'<<'*20}")
