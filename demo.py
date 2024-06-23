import os
import sys
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.exception import CreditException
from Give_Me_Some_Credit.config.configuration import Configuration
from Give_Me_Some_Credit.pipeline.training_pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    try:
        config_path = os.path.join("config", "config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        pipeline.start()
        logging.info("main function is executing.")
    except Exception as e:
        raise CreditException(e, sys) from e


if __name__ == "__main__":
    main()
