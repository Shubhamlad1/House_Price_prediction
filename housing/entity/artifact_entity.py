from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", 
["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact= namedtuple(
    "DataValidationArtifact", 
    ["schema_file_path","report_file_path","report_page_file_path","is_validate","message"]
)

DataTransformationArtifact= namedtuple(
    "DataTransformationArtifact",
    ["is_transformed", "message", "processed_object_file_path", "transformed_test_file_path", "transformed_train_file_path"]
)

ModelTrainerArtifact= namedtuple(
    "ModelTrainerArtifact",
    ["is_trasformed", "message", "trained_model_file_path",
    "train_rsme", "test_rsme","train_accuracy", "test_accuracy", "model_accuracy"]
)

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])

