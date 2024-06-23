import os
import sys
import json
import pandas as pd
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.util.util import read_yaml_file
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.entity.config_entity import (
    DataValidationConfig,
    DataIngestionConfig,
)
from Give_Me_Some_Credit.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from evidently.dashboard import Dashboard
from evidently.model_profile import Profile
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile.sections import DataDriftProfileSection


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_ingestion_config: DataIngestionConfig,
    ):
        try:
            logging.info(f"{'>>'*20} Data Validation Started {'<<'*20}.")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_ingestion_config = data_ingestion_config
            self.schema_file_path = self.data_validation_config.schema_file_path
            self.dataset_schema = read_yaml_file(file_path=self.schema_file_path)
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_train_and_test_df(self):
        try:
            logging.info("Reading the Training and Testing CSV Files.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
            logging.info("Reading the Training and Testing CSV Files Done.")
            return train_df, test_df
        except Exception as e:
            raise CreditException(e, sys) from e

    def detect_train_and_test_files(self):
        try:
            logging.info("Checking whether training and test files are present.")
            is_train_file_present = False
            is_test_file_present = False
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            is_train_file_present = os.path.exists(train_file_path)
            is_test_file_present = os.path.exists(test_file_path)
            are_available = is_train_file_present and is_test_file_present
            logging.info(
                f"Do the training and testing files exist? -> {are_available}."
            )
            if not are_available:
                train_file = self.data_ingestion_artifact.train_file_path
                test_file = self.data_ingestion_artifact.test_file_path
                message = f"Either Training file: {train_file} or Testing file: {test_file} is not present."
                raise Exception(message)
            return are_available
        except Exception as e:
            raise CreditException(e, sys) from e

    def check_columns(self, file_path, file):
        try:
            logging.info(
                f"Validating the number of columns present in the file as per \
                schema.yaml and columns that have missing values present or not in {file}."
            )
            df = pd.read_csv(file_path)
            num_cols = df.shape[1]
            cols = list(df.columns)
            num_cols = len([col for col in cols if col not in ["Unnamed: 0"]])
            if num_cols != self.dataset_schema["num_cols"]:
                raise Exception(
                    f"Number of features/columns in {file} are different from schema.yaml file."
                )

            cols = [
                col
                for col in cols
                if col not in [self.dataset_schema["target_column"], "Unnamed: 0"]
            ]
            for col in cols:
                if col not in self.dataset_schema["columns"].keys():
                    raise Exception(f"{col} in {file} is Absent in schema.")
            num_cols_with_missing_values = 0
            cols_with_missing_values = []
            columns = df.columns.tolist()
            for col in columns:
                if df[col].isnull().sum():
                    num_cols_with_missing_values += 1
                    cols_with_missing_values.append(col)
            logging.info(
                f"Columns that have missing values present are - {cols_with_missing_values}"
            )
            logging.info(
                "Checking number of columns and missing values in every column Done!."
            )
            return True
        except Exception as e:
            raise CreditException(e, sys) from e

    def validate_dataset_schema(self):
        try:
            logging.info("Validating the dataset schema.")
            validation_status = False
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            for file in os.listdir(raw_data_dir):
                if self.check_columns(os.path.join(raw_data_dir, file), file):
                    validation_status = True
            return validation_status
        except Exception as e:
            raise CreditException(e, sys) from e

    def get_and_save_data_drift_report(self):
        try:
            logging.info("Generating Data Drift report file.")
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df, test_df = self.get_train_and_test_df()
            profile.calculate(train_df, test_df)
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            logging.info("Report generated successfully.")
            return report
        except Exception as e:
            raise CreditException(e, sys) from e

    def save_data_drift_report_page(self):
        try:
            logging.info("Generating Data Drift report page.")
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_df()
            dashboard.calculate(train_df, test_df)
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)
            dashboard.save(report_page_file_path)
            logging.info("Data Drift report page generated successfully.")
        except Exception as e:
            raise CreditException(e, sys) from e

    def is_data_drift_found(self):
        try:
            logging.info("Checking Data drift")
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_data_validation(self):
        try:
            logging.info("Initiating Data Validation.")
            self.detect_train_and_test_files()
            self.validate_dataset_schema()
            self.is_data_drift_found()
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successfully.",
            )
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20} Data Validation completed {'<<'*20}.")
