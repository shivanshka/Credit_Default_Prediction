from typing import List
from setuptools import setup, find_packages

PROJECT_NAME = "Give Me Some Credit"
AUTHOR = "Chirag Sharma"
VERSION = "0.0.1"
DESCRIPTION = "This is a Machine learning project to Improve on the \
        state of the art in credit scoring by predicting the probability \
        that somebody will experience financial distress in the next two years."

REQUIREMENT_FILE_NAME = "requirements.txt"

HYPHEN_E_DOT = "-e ."


def get_requirements_list(filename=REQUIREMENT_FILE_NAME) -> List[str]:
    with open(filename) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [
            requirement_name.replace("\n", "") for requirement_name in requirement_list
        ]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list


setup(
    name=PROJECT_NAME,
    author=AUTHOR,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list(filename=REQUIREMENT_FILE_NAME),
)
