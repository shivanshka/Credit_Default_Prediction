import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from Give_Me_Some_Credit.constants import *
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.util.util import read_yaml_file, save_data, save_object
from Give_Me_Some_Credit.util.preprocessors import (
    OutlierTreatment,
    RareLabelCategoricalEncoder,
    CustomPowerTransformer,
    TotalNumberOfDues,
    DropUnecessaryFeatures,
)
from Give_Me_Some_Credit.entity.config_entity import DataTransformationConfig
from Give_Me_Some_Credit.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            logging.info(f"{'>>'*20} Data Transformation has started {'<<'*20}.")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            self.data_schema = read_yaml_file(file_path=self.schema_file_path)
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_data_transformer_object(self):
        try:
            logging.info("Creating Pipeline Object.")
            pipeline_obj = Pipeline(
                [
                    (
                        "age_outlier_treatment",
                        OutlierTreatment(
                            variable=self.data_schema[OUTLIER_REMOVAL_VARS_KEY][0],
                            upper_quantile=AGE_UPPER_QUANTILE,
                            lower_quantile=AGE_LOWER_QUANTILE,
                        ),
                    ),
                    (
                        "rare_label_encoding",
                        RareLabelCategoricalEncoder(
                            variables=self.data_schema[
                                RARE_LABEL_VARS_WITH_SAME_TOLERANCE_KEY
                            ],
                            tolerance=RARE_LABEL_VARS_TOLERANCE,
                        ),
                    ),
                    (
                        "cube_root_of_age",
                        CustomPowerTransformer(
                            variable=self.data_schema[OUTLIER_REMOVAL_VARS_KEY][0],
                            power=CUBEROOT_AGE_POWER,
                        ),
                    ),
                    (
                        "cube_root_of_revolving_utilization_of_unsecured_lines",
                        CustomPowerTransformer(
                            variable=self.data_schema[
                                CUBE_ROOT_OF_REVOLVING_UTILIZATION_OF_UNSECURED_LINES_KEY
                            ][0],
                            power=CUBEROOT_REVOLVING_UTILIZATION_OF_UNSECURED_LINES_POWER,
                        ),
                    ),
                    (
                        "extracting_total_number_dues",
                        TotalNumberOfDues(
                            variables=self.data_schema[TOTAL_NUMBER_OF_DUES_KEY]
                        ),
                    ),
                    (
                        "dropping_age",
                        DropUnecessaryFeatures(
                            variables_to_drop=self.data_schema[
                                OUTLIER_REMOVAL_VARS_KEY
                            ][0]
                        ),
                    ),
                ]
            )
            logging.info("Pipeline Object Created Successfully.")
            return pipeline_obj
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Obtaining training & testing file paths")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Reading training and testing CSV files")
            train_data = pd.read_csv(filepath_or_buffer=train_file_path)
            test_data = pd.read_csv(filepath_or_buffer=test_file_path)

            logging.info("Dropping columns that are not in USE_COLS list.")
            cols_to_drop = [
                col
                for col in train_data.columns
                if col not in self.data_schema[USE_COLS_KEY]
            ]
            train_data.drop(columns=cols_to_drop, axis=1, inplace=True)
            test_data.drop(columns=cols_to_drop, axis=1, inplace=True)

            logging.info("Capturing X & Y features for train & test dataset.")
            x_train = train_data.drop(self.data_schema["target_column"], axis=1)
            y_train = train_data[self.data_schema["target_column"]]
            x_test = test_data.drop(self.data_schema["target_column"], axis=1)
            y_test = test_data[self.data_schema["target_column"]]

            logging.info(
                "Applying preprocessing/pipeline object on training & testing X features."
            )
            pipeline = self.get_data_transformer_object()
            train_pipeline_result = pipeline.fit_transform(x_train)
            test_pipeline_result = pipeline.transform(x_test)

            logging.info("Concatenating training & testing X & Y features.")
            transformed_train_data = pd.concat([train_pipeline_result, y_train], axis=1)
            transformed_test_data = pd.concat([test_pipeline_result, y_test], axis=1)

            transformed_train_dir = (
                self.data_transformation_config.transformed_train_dir
            )
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            transformed_train_file_path = os.path.join(
                transformed_train_dir, "transformed_train.csv"
            )
            transformed_test_file_path = os.path.join(
                transformed_test_dir, "transformed_test.csv"
            )

            logging.info("Saving preprocessing training and testing data.")
            save_data(
                file_path=transformed_train_file_path, data=transformed_train_data
            )
            save_data(file_path=transformed_test_file_path, data=transformed_test_data)

            logging.info("Saving preprocessing pipeline object.")
            preprocessing_object_file_path = (
                self.data_transformation_config.preprocessed_object_file_path
            )
            save_object(file_path=preprocessing_object_file_path, obj=pipeline)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_object_file_path,
                is_transformed=True,
                message=f"Data Transformation completed successfully.",
            )
            logging.info(f"Data Transformation: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"\n{'>>' * 30}Data Transformation log completed.{'<<' * 30}\n\n")
