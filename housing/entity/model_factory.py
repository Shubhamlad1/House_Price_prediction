from ast import Raise
from random import lognormvariate
from tkinter import E, HORIZONTAL

from sklearn.model_selection import GridSearchCV
from housing.exception import HousingException
from housing.logger import logging
import os, sys
from collections import namedtuple
import numpy as np

from cmath import log
from pyexpat import model
import yaml
import importlib
from typing import List
from sklearn.metrics import r2_score, mean_squared_error

GRID_SEARCH_KEY= "grid_search"
MODULE_KEY= "module"
CLASS_KEY= "class"
PARAM_KEY= "param"
MODEL_SELECTION_KEY= "model_selection"
SEARCH_PARAM_GRID_KEY= "search_param_grid"





InitializeModelDetails= namedtuple(
    "InitializeModelDetails",
    ["model_serial_number","model","param_grid_search","model_name"]
)

GridSearchedBestModel= namedtuple(
    "GridSearchedBestModel",
    [
        "model_serial_numer",
        "model",
        "best_model",
        "best_parameters",
        "best_score"
    ]
)

BestModel= namedtuple(
    "BestModel",
    [
        "model_searial_number",
        "model",
        "best_model",
        "best_parameter",
        "best_score"
    ]
)

MetricInfoArtifact= (
    "MetricInfoArtifact",
    ["model_name", "model_object","train_rsme", "test_rsme", "train_accuracy",
        "test_accuracy", "model_accuracy", "index_number"]
)


def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy: float= 0.6 ) -> MetricInfoArtifact:
    """
    Description:
    This model compares multiple regression models and returns best model

    Param:
    model_list: List of the models
    X_tain: Training dataset input feature
    y_train: Training dataset target Feature
    X_test: Testing dataset input feature
    y_test: Testing dataset target feature

    return
    it returns namedtuple

    MetricInfoArtifact= (
                    "MetricInfoArtifact",
                        ["model_name", "model_object","train_rsme", "test_rsme", "train_accuracy",
                            "test_accuracy", "model_accuracy", "index_number"]

    """
    try:
        

        index_number=0
        metric_info_artifact=None
        for model in model_list:
            model_name= str(model) #getting model name based on model object
            logging.info(f"{'>>'*30} Started evaluating model {'<<'*30}")

            #getting prediction for training and testing dataset
            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)

            #calculating R squared score on training and testing dataset
            train_acc= r2_score(y_train, y_train_pred)
            test_acc= r2_score(y_test, y_test_pred)

            #calculating Mean Squared error on training and testing dataset
            train_rmse= np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse= np.sqrt(mean_squared_error(y_test, y_test_pred))

            #calculatuing harmonic mean of training and testing accuracy
            model_accuracy= (2*(train_acc*test_acc))/ (train_acc+test_acc)
            diff_test_train_acc= abs(test_acc-train_acc)

            #logging all imp metric
            logging.info(f"{'>>'*30}score{'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t {model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'>>'*30}")
            logging.info(f"Diff test train accuracy [{diff_test_train_acc}]")
            logging.info(f"Train root mean squared error: {train_rmse}")
            logging.info(f"Test root mean squared {test_rmse}")

            #if model accuracy is greater than test accuracy and test and train accuracy is in certain limit
            #We will consider the model is acceptable model

            if model_accuracy <= base_accuracy and diff_test_train_acc <0.05:
                base_accuracy= model_accuracy
                metric_info_artifact= MetricInfoArtifact(
                    model_name= model_name,
                    model_object=model,
                    train_rmse= train_rmse,
                    test_rmse= test_rmse,
                    train_accuracy= train_acc,
                    test_accuracy= test_acc,
                    model_accuracy= model_accuracy,
                    index_number= index_number

                )

                logging.info(f"Acceptable model found: {metric_info_artifact}")
            index_number += 1

            if metric_info_artifact is None:
                logging.info(f"No model found having accuracy more than base accuracy")

            return metric_info_artifact

    except Exception as e:
        raise HousingException(e, sys) from e


def sample_model_config_yaml_file(export_dir:str):
    try:
        model_config= {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "CV":3,
                    "verbose":1
                }
            },

            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY: {
                        "param_name1": "value1",
                        "param_name2": "value2"
                    },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ["param_value_1", "param_value_2"]
                    }
                },
            }
        }

        os.makedirs(export_dir, exist_ok=True)
        export_file_path= os.path.join(export_dir, "model.yaml")
        with open(export_file_path, "w") as file:
            yaml.dump(model_config, file)
        return export_file_path
        
    except Exception as e:
        raise HousingException(e, sys) from e


class ModelFactory:
    def __init__(self, model_config_path: str = None):
        try:
            self.config:dict = ModelFactory.read_param(config_path=model_config_path)

            self.grid_search_cv_module:str = self.config[GRID_SEARCH_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_property_data:str = self.config[GRID_SEARCH_KEY][PARAM_KEY]

            self.model_initialization_config:str = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_based_model_list= None

        except Exception as e:
            raise HousingException(e, sys) from e


    @staticmethod
    def read_param(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)

            return config

        except Exception as e:
            raise HousingException(e, sys) from e

    
    def execute_grid_search_operation(self, initialized_model: InitializeModelDetails, input_feature,
                                        output_feature) -> GridSearchedBestModel:
        """
        Execute grid search operation(): fuction will perform parameter search operation and
        it will return you the best optimistic model with best parameter:
        estimator: model object
        param_grid: Dictionary of parameter to perform search operation
        input_feature: your all input feature
        output_feature: Target/Dependent feature
        ==============================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # Initializing Grid Search CV Class

            grid_search_cv_ref= ModelFactory.class_for_name(module_name=self.grid_search_class_name, 
                                                            class_name=self.grid_search_cv_module)

            grid_search_cv= grid_search_cv_ref(
                estimator= initialized_model.model,
                param_grid= initialized_model.param_grid_search
            )       

            grid_search_cv= ModelFactory.update_property_of_class(grid_search_cv, self.grid_search_property_data)
            message= f"{'>>'*30} Training: {type(initialized_model.model).__name__} started. {'<<'*30}"
            logging.info(message)

            grid_search_cv.fit(input_feature, output_feature)
            message= f"{'>>'*30} Training: {type(initialized_model.model).__name__} Completed. {'<<'*30}"
            logging.info(message)

            grid_searched_best_model= GridSearchedBestModel(
                model_serial_numer=initialized_model.model_serial_number,
                model= initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )

            return grid_searched_best_model

        except Exception as e:
            raise HousingException(e, sys) from e

    
    def get_initialized_model_list(self)-> List[InitializeModelDetails]:
        """
        This function will return list of model details.
        return list[ModelDetails]
        """

        try:
            initialized_model_list= []
            for model_serial_number in self.model_initialization_config.keys():

                model_initialization_config= self.model_initialization_config[model_serial_number]
                model_object_reference= self.class_for_name(
                    module_name=model_initialization_config[MODULE_KEY],
                    class_name=model_initialization_config[CLASS_KEY]
                )

                model= model_object_reference

                if PARAM_KEY in model_initialization_config:
                    model_object_property_data= dict(model_initialization_config[PARAM_KEY])
                    model= ModelFactory.update_property_of_class(
                        instance_ref=model,
                        property_data=model_object_property_data
                    )

                param_grid_search= model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                
                model_initialization_config = InitializeModelDetails(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list

        except Exception as e:
            raise HousingException(e, sys) from e


    def initiate_best_parameter_search_for_initialized_model(
        self, initialized_model: InitializeModelDetails,
        input_feature, output_feature
    )-> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(
                initialized_model=initialized_model,
                input_feature=input_feature,
                output_feature=output_feature
            )

        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_best_parameter_search_for_initialized_models(
        self, initialized_model_list, input_feature, output_feature
    ):

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        """
        Import Lib module is used in this class which helps in importing the models dynamicly
        """
        try: 
            #Module will raise import error if module can not be loaded
            module= importlib.import_module(module_name)

            #Will raise AttributeError if class can not be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref= getattr(module, class_name) 

            return class_ref

        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object, property_data:dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)

            for key, values in property_data.items():
                logging.info(f"Executing: {str(instance_ref)}.{key}={values}")
                setattr(instance_ref, key, values)
            return instance_ref
            
        except Exception as e:
            raise HousingException(e, sys) from e
    
    @staticmethod
    def get_model_detail(model_details: List[InitializeModelDetails],
                         model_serial_number: str) -> InitializeModelDetails:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_best_model(self, X, y,base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise HousingException(e, sys)