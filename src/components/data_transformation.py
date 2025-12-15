import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_objects

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifact","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_feature=['writing score','reading score']
            categorical_feature=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical column: {num_feature}")
            logging.info(f"Categorical columns: {categorical_feature}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_feature),
                    ("cat_pipeline",cat_pipeline,categorical_feature)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column="math score"
            num_feature=['writing score','reading score']
            
            input_feature_train=df_train.drop(columns=[target_column],axis=1)
            target_feature_train=df_train[target_column]

            input_feature_test=df_test.drop(columns=[target_column],axis=1)
            target_feature_test=df_test[target_column]


            logging.info("Applying preproceer object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr=np.c_[
                    input_feature_train_arr,np.array(target_feature_train)
            ]
            test_arr=np.c_[
                    input_feature_test_arr,np.array(target_feature_test)
            ]
            logging.info("Saved processing object")

            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
                    