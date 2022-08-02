from tkinter import E
import yaml
from housing.constants import DATASET_SCHEMA_COLUMNS_KEY
from housing.exception import HousingException
import os,sys
import pandas as pd
import numpy as np
import dill

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise HousingException(e,sys)

def read_yaml_file(file_path:str) -> dict:
    """
    This fuction will read YAML file in the form of dictionary
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e, sys) from e 


def load_data(file_path: str, schema_file_path: str):
    try:
        dataset_schema= read_yaml_file(schema_file_path)

        schema= dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe= pd.read_csv(file_path)

        error_msg= ""

        for column in dataframe.columns:
            if column in list(schema.keys):
                dataframe[column].astype(schema[column])

        else:
            error_msg= f"{error_msg} \nColumn: [{column}] is not in the schema"

        if len(error_msg) > 0:
            raise Exception(error_msg)

        return dataframe
    except Exception as e:
        raise HousingException(e,sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array at file path and array is the np.array to save
    """
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            np.save(file_object, array)

    except Exception as e:
        raise HousingException(e,sys) from e


def save_object(file_path: str, object):
    """
    saves any sort of object at file path= file_path
    """
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj=object, file=file_obj)

    except Exception as e:
        raise HousingException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise HousingException(e, sys) from e

def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise HousingException(e,sys) from e