import os
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from Give_Me_Some_Credit.logger import logging
from Give_Me_Some_Credit.exception import CreditException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditException(e, sys) from e


def save_data(file_path: str, data: pd.DataFrame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(file_path, index=None)
    except Exception as e:
        raise CreditException(e, sys) from e


def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CreditException(e, sys) from e


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CreditException(e, sys) from e


def write_yaml_file(file_path: str, data: dict = None):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise CreditException(e, sys) from e
