from setuptools import setup, find_packages
from typing import List

def get_requirements_list()->List[str]:
    """
    Description: This function will write the list of requiremets mentioned
    in the requirements.txt file.

    All names of libraries mentioned in the requirements.txt file will be 
    listed in a List and in a string format

    """
    with open("requirements.txt") as requiremets_file:
        return requiremets_file.readlines().remove("-e .")


setup(
name="housing-predictor",
version="0.0.3",
author="Shubham Lad",
description="Machine learning model for house price prediction",
packages=find_packages(),
install_requires=get_requirements_list()

)




