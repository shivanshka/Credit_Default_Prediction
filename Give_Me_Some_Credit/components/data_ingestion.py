import os
import sys
import requests
import pandas as pd
from pyunpack import Archive

from Give_Me_Some_Credit.logger import logging
from sklearn.model_selection import StratifiedShuffleSplit
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.entity.artifact_entity import DataIngestionArtifact 
from Give_Me_Some_Credit.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion has started {'<<'*20}.")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditException(e, sys) from e

    def download_data(self):
        try:
            download_url = self.data_ingestion_config.dataset_download_url
            zipped_download_dir = self.data_ingestion_config.zipped_download_dir
            os.makedirs(zipped_download_dir, exist_ok=True)
            credit_file_name = os.path.basename(download_url)
            zipped_file_path = os.path.join(zipped_download_dir, credit_file_name)
            logging.info(
                f"{'>>'*20} Downloading the dataset.rar file from {download_url} into {zipped_file_path}."
            )
            r = requests.get(download_url, allow_redirects=True)
            open(zipped_file_path, "wb").write(r.content)
            logging.info(
                f"File: {[zipped_file_path]} has been downloaded successfully."
            )
            return zipped_file_path
        except Exception as e:
            raise CreditException(e, sys) from e

    def extract_zipped_files(self, zipped_file_path: str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Extracting the dataset.rar file into {raw_data_dir}.")
            Archive(zipped_file_path).extractall(raw_data_dir)
            logging.info(f"Extraction finished at {raw_data_dir}.")
        except Exception as e:
            raise CreditException(e, sys) from e

    def split_data_into_train_and_test(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            original_data = os.path.join(raw_data_dir, file_name)
            logging.info(f"Reading CSV File: {[original_data]}")
            credit_df = pd.read_csv(original_data)
            logging.info("Splitting the original dataframe into train & test files.")
            strat_train_split = None
            strat_test_split = None
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(
                credit_df, credit_df["SeriousDlqin2yrs"]
            ):
                strat_train_split, strat_test_split = (
                    credit_df.iloc[train_index],
                    credit_df.iloc[test_index],
                )
            train_file_path = os.path.join(
                self.data_ingestion_config.ingested_train_dir, file_name
            )
            test_file_path = os.path.join(
                self.data_ingestion_config.ingested_test_dir, "test.csv"
            )

            if strat_train_split is not None:
                os.makedirs(
                    self.data_ingestion_config.ingested_train_dir, exist_ok=True
                )
                logging.info(f"Exporting training dataset to {train_file_path}")
                strat_train_split.to_csv(train_file_path, index=False)

            if strat_test_split is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting testing dataset to {test_file_path}")
                strat_test_split.to_csv(test_file_path, index=False)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message="Data Ingestion done successfully.",
            )
            logging.info(f"Data Ingestion artifact {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            zipped_file_path = self.download_data()
            self.extract_zipped_files(zipped_file_path=zipped_file_path)
            return self.split_data_into_train_and_test()
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20} Data Ingestion Completed {'<<'*20}.")
