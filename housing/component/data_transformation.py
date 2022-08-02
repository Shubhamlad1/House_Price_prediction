import sklearn
from sklearn import preprocessing
from build.lib import housing
from housing.constants import *
from housing.exception import HousingException
from housing.logger import logging
from housing.config.configuration import Configuration
from housing.entity.config_entity import DataTransformationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
import os, sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from housing.util.util import read_yaml_file, load_data, save_numpy_array_data, save_object
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedroom_per_room=True,
                    total_rooms_ix=3,
                    population_ix=5,
                    household_ix=6,
                    total_bedrooms_ix=4, columns=None):
        """
        FeatureGenerator Initialization
        add_bedroom_per_room: bool
        total_room_ix: Int index number of total rooms columns
        population_ix: Int index number of total population coumns
        household_ix: Int index number of household columns
        total_bedrooms_ix: Int index number of bedroom columns
        """

        try:
            self.columns= columns
            if self.columns is not None:
                total_rooms_ix= self.columns.index(COLUMN_TOTAL_ROOMS)
                population_ix=self.columns.index(COLUMN_POPULATION)
                households_ix=self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_ix=self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.add_bedroom_per_room= add_bedroom_per_room
            self.total_room_ix= total_rooms_ix
            self.population_ix= population_ix
            self.household_ix= household_ix
            self.total_bedrooms_ix= total_bedrooms_ix

        except Exception as e:
            raise HousingException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            room_per_households= X[:, self.total_room_ix]/ X[:, self.household_ix]
            population_per_household= X[:,self.population_ix] / X[:, self.household_ix]

            if self.add_bedroom_per_room:
                bedrooms_per_room= X[:, self.total_bedrooms_ix] / X[:, self.total_room_ix]

                generated_feature= np.c_[
                    X, room_per_households, population_per_household, bedrooms_per_room
                ]

            else:
                generated_feature= np.c_[
                    X, room_per_households, population_per_household
                ]

            return generated_feature

        except Exception as e:
            raise HousingException(e,sys) from e


class DataTransformation:

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation Log Started {'<<' * 30}")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact= data_ingestion_artifact
            self.data_validation_artifact= data_validation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_data_transformation_object(self)-> ColumnTransformer:
        try:
            schema_file_path= self.data_validation_artifact.schema_file_path
            dataset_schema= read_yaml_file(schema_file_path)

            numerical_columns= dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns= dataset_schema[CATEGORICAL_COLUMN_KEY]

            num_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    (
                        'feature_generator', FeatureGenerator(
                            add_bedroom_per_room=self.data_transformation_config.add_bedroom_per_room,
                            columns= numerical_columns
                        )
                    ),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline= Pipeline([
                ('imputer', SimpleImputer(strategy= "most_frequent")),
                (
                    'one_hot_encoder', OneHotEncoder()
                ),
                ('Scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical_Columns: {categorical_columns}")
            logging.info(f"Numerical_Columns: {numerical_columns}")

            preprocessing= ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, cat_pipeline)
                ]
            )

            return preprocessing

        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining Preprocessing Objects")
            preprocessing_obj= self.get_data_transformation_object()

            logging.info(f"Obtaining Training and Test File")
            train_file_path= self.data_ingestion_artifact.train_file_path
            test_file_path=  self.data_ingestion_artifact.test_file_path

            schema_file_path= self.data_validation_artifact.schema_file_path

            logging.info("Loading Train and Test data as pandas DataFrame")
            train_df= load_data(file_path=train_file_path, schema_file_path=schema_file_path)

            test_df= load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema= read_yaml_file(file_path=schema_file_path)
            target_column_name= schema[TARGET_COLUMN_KEY]

            logging.info(f"spliting input and target features from training and testing dataset")
            input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info("applying preprocessing object on training and testing dataframe")
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            transformed_train_dir= self.data_transformation_config.transformaed_train_dir
            transformed_test_dir= self.data_transformation_config.transformed_test_dir

            train_file_name= os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name= os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path= os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path= os.path.join(transformed_test_dir, test_file_name)

            logging.info("Saving testing and training arry")

            save_numpy_array_data(file_path=transformed_train_dir, array= train_arr)
            save_numpy_array_data(file_path=transformed_test_dir, array= test_arr)

            processing_object_file_path= self.data_transformation_config.preprocessed_object_file_path

            logging.info("Saving Processing object")
            save_object(file_path=processing_object_file_path, object=preprocessing_obj)

            data_transformation_artifact= DataTransformationArtifact(
                is_transformed= True,
                message= "data Transformation completed",
                processed_object_file_path=processing_object_file_path,
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path
            )

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact


        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30} Data Trnasformation Completed {'<<'*30}\n\n")




