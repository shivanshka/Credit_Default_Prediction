import os
import sys
import yaml
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from Give_Me_Some_Credit.constants import *
from Give_Me_Some_Credit.logger import logging
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.entity.artifact_entity import MetricsInfoArtifact
from Give_Me_Some_Credit.entity.config_entity import (
    BestModel,
    GridSearchBestModel,
    InitializedModelDetail,
)


def search_optimum_threshold(y_test, y_pred):
    try:
        logging.info("Searching optimum threshold.")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        J = tpr - fpr
        ix = np.argmax(J)
        best_threshold = thresholds[ix]
        logging.info(f"Optimum threshold found: {float(best_threshold)}")
        return float(best_threshold)
    except Exception as e:
        raise CreditException(e, sys) from e


def evaluate_classification_model(
    model_list, X_train, y_train, X_test, y_test, base_auc=0.5
):
    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)
            logging.info(
                f"{'>>'*10} Started evaluating model: [{type(model).__name__}] {'<<'*10}."
            )

            y_train_pred = model.predict_proba(X_train)[:, 1]
            y_test_pred = model.predict_proba(X_test)[:, 1]
            threshold = search_optimum_threshold(y_test, y_test_pred)

            train_auc_score = roc_auc_score(y_true=y_train, y_score=y_train_pred)
            test_auc_score = roc_auc_score(y_true=y_test, y_score=y_test_pred)

            train_log_loss = log_loss(y_true=y_train, y_pred=y_train_pred)
            test_log_loss = log_loss(y_true=y_test, y_pred=y_test_pred)

            abs_train_test_log_loss_diff = abs(train_log_loss - test_log_loss)

            logging.info(f"{'>>'*10} Loss {'<<'*10}.")
            logging.info(f"Train auc score\t\t Test auc score.")
            logging.info(f"{train_auc_score}\t\t {test_auc_score}.")

            logging.info(f"{'>>'*10} Loss {'<<'*10}.")
            logging.info(f"Train log-loss\t\t Test log-loss.")
            logging.info(f"{train_log_loss}\t\t {test_log_loss}.")

            if (
                train_auc_score >= base_auc
                and test_auc_score >= base_auc
                and abs_train_test_log_loss_diff < 0.02
            ):
                metric_info_artifact = MetricsInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_auc_score=train_auc_score,
                    test_auc_score=test_auc_score,
                    train_log_loss=train_log_loss,
                    test_log_loss=test_log_loss,
                    index_number=index_number,
                    threshold=threshold,
                )
            logging.info(f"Acceptable model found {metric_info_artifact}.")
            index_number += 1
        if metric_info_artifact is None:
            logging.info("No model found with higher auc than base_auc.")
        return metric_info_artifact
    except Exception as e:
        raise CreditException(e, sys) from e


class ModelFactory:
    def __init__(self, model_config_path):
        try:
            self.config = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data = dict(
                self.config[GRID_SEARCH_KEY][PARAMS_KEY]
            )
            self.models_initialization_config = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_list = None
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref, property_data):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property data paramater required is dictionary.")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing: {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def read_params(config_path):
        try:
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
                return config
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise CreditException(e, sys) from e

    def execute_grid_search_operation(
        self, initialized_model, input_feature, output_feature
    ):
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(
                module_name=self.grid_search_cv_module,
                class_name=self.grid_search_class_name,
            )
            grid_search_cv = grid_search_cv_ref(
                estimator=initialized_model.model,
                param_grid=initialized_model.param_grid_search,
            )
            grid_search_cv = ModelFactory.update_property_of_class(
                grid_search_cv, self.grid_search_property_data
            )
            message = f'{">>"* 10} f"Training {type(initialized_model.model).__name__} Started." {"<<"*10}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>"* 10} f"Training {type(initialized_model.model).__name__}" completed {"<<"*10}'
            grid_searched_best_model = GridSearchBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_,
            )
            return grid_searched_best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_initialized_model_list(self):
        try:
            initiaized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[
                    model_serial_number
                ]
                model_obj_ref = ModelFactory.class_for_name(
                    module_name=model_initialization_config[MODULE_KEY],
                    class_name=model_initialization_config[CLASS_KEY],
                )
                model = model_obj_ref()
                # model = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)

                if PARAMS_KEY in model_initialization_config:
                    model_obj_property_data = dict(
                        model_initialization_config[PARAMS_KEY]
                    )
                    model = ModelFactory.update_property_of_class(
                        instance_ref=model, property_data=model_obj_property_data
                    )
                param_grid_search = model_initialization_config[SEARCH_PARAM_KEY_GRID]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                model_initialization_config = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model=model,
                    param_grid_search=param_grid_search,
                    model_name=model_name,
                )
                initiaized_model_list.append(model_initialization_config)
            self.initialized_model_list = initiaized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initalized_model(
        self, initialized_model, input_feature, output_feature
    ):
        try:
            return self.execute_grid_search_operation(
                initialized_model=initialized_model,
                input_feature=input_feature,
                output_feature=output_feature,
            )
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_best_parameter_search_for_initalized_models(
        self, initialized_model_list, input_feature, output_feature
    ):
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list_ in initialized_model_list:
                grid_searched_best_model = (
                    self.initiate_best_parameter_search_for_initalized_model(
                        initialized_model=initialized_model_list_,
                        input_feature=input_feature,
                        output_feature=output_feature,
                    )
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details, model_serial_number):
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise CreditException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(
        grid_searched_best_model_list, base_auc=0.5
    ):
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_auc < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of the model has base auc score: {base_auc}.")
            logging.info(f"Best model: {best_model}.")
            return best_model
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_best_model(self, X, y, base_auc=0.5):
        try:
            logging.info("Started initializing model from config file.")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = (
                self.initiate_best_parameter_search_for_initalized_models(
                    initialized_model_list=initialized_model_list,
                    input_feature=X,
                    output_feature=y,
                )
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_auc=base_auc,
            )
        except Exception as e:
            raise CreditException(e, sys) from e
