import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
import logging
import sys
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

sys.setrecursionlimit(10**6)

class TopicModelingPipeline:
    def __init__(self, partition_models,min_similarity=0.7):
        self.min_similarity = min_similarity
        self.partition_models = partition_models

        self.log_folder = f"../../job_outputs/merged_model_update"

        # Ensure log folder exists
        os.makedirs(self.log_folder, exist_ok=True)

        self.merged_model = None
        self.models = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Log to file
        log_file = os.path.join(self.log_folder, f"pipeline.log")  # Define log file path
        fh = logging.FileHandler(log_file)  # File handler for logging to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)  # Add file handler to logger

    def load_model(self, model_path):
        """
        Load a BERTopic model from the specified file path.

        Args:
            model_path (str): The file path to the model to be loaded.

        Returns:
            BERTopic: The loaded BERTopic model if successful, None otherwise.

        Logs:
            Info: Logs the start and successful completion of the model loading process.
            Error: Logs any errors encountered during the model loading process.
        """
        try:
            self.logger.info(f"Loading model from: {model_path}")
            model = BERTopic.load(model_path, embedding_model=self.sentence_model)
            self.logger.info(f"Model {os.path.basename(model_path)} loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def merge_models_incrementally(self):
        """
        Merges topic models incrementally in batches to manage memory usage.
        This method merges topic models stored in two different folders incrementally.
        The merging process is done in batches to avoid memory overflow. The method 
        logs the progress and any errors encountered during the merging process.
        The merging process is divided into two parts:
        1. Merging models from `self.partition_models` in batches of size `batch_size`.
        The merged model is stored in `self.merged_model`.
        Attributes:
            batch_size (int): The number of models to merge in each batch.
            self.partition_models (str): The directory containing the first set of models.
            self.merged_model (BERTopic): The resulting merged model.
            self.min_similarity (float): The minimum similarity threshold for merging models.
            self.logger (Logger): Logger for logging information and errors.
        Raises:
            Exception: If an error occurs during the merging process, it is logged.
        """

        try:
            self.logger.info("Merging models incrementally")

            batch_size = 4  # Adjust this based on available memory

            for i in range(1, 11, batch_size):
                batch_models = []
                for j in range(i, min(i + batch_size, 11)):
                    model_name = f"topic_model_{j}"
                    model_path = os.path.join(self.partition_models, model_name)
                    model = self.load_model(model_path)
                    if model:
                        batch_models.append(model)

                if batch_models:
                    self.logger.info(f"Merging models: {', '.join([f'topic_model_{j}' for j in range(i, min(i + batch_size, 11))])}")
                    if self.merged_model is None:
                        self.merged_model = batch_models[0]
                        batch_models = batch_models[1:]
                    
                    self.merged_model = BERTopic.merge_models(
                        [self.merged_model] + batch_models,
                        min_similarity=self.min_similarity
                    )
                    # Optionally delete the models after merging to free memory
                    del batch_models
                    self.logger.info(f"Models merged successfully up to topic_model_{min(i + batch_size - 1, 7)}")

        except Exception as e:
            self.logger.error("Error merging models: {}".format(e))
            
    def save_topic_model(self, save_path):
        try:
            self.logger.info("Saving BERTopic model to: {}".format(save_path))
            self.merged_model.save(save_path, serialization='safetensors', 
                                  save_embedding_model=self.sentence_model)
        except Exception as e:
            self.logger.error("Error saving BERTopic model: {}".format(e))
    
    def save_topic_info(self, file_path):
        """
        Saves the topic information to a CSV file.

        Parameters:
        file_path (str): The path to the file where the topic information will be saved.

        This method retrieves the topic information using the `get_topic_info` method
        and saves it to the specified file path in CSV format. If an error occurs during
        the process, it logs an error message.

        Raises:
        Exception: If there is an error while saving the topic information.
        """
        try:
            self.logger.info("Saving topic info to file: {}".format(file_path))
            topic_info = self.get_topic_info()
            topic_info.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error("Error saving topic info: {}".format(e))

    def run_pipeline(self):
        try:
            model_save_path = os.path.join(self.log_folder)
            topic_info_file = os.path.join(self.log_folder, f"merged_topic_info.csv")

            self.logger.info("Starting pipeline")
            self.sentence_model = SentenceTransformer("jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br")
            self.merge_models_incrementally()
            self.save_topic_model(model_save_path)
            self.save_topic_info(topic_info_file)
            self.logger.info("Pipeline completed successfully. Model saved in: {}".format(model_save_path))
        except Exception as e:
            self.logger.error("Error in pipeline execution: {}".format(e))


    def get_topic_info(self):
        try:
            self.logger.info("Getting topic info")
            return self.merged_model.get_topic_info()
        except Exception as e:
            self.logger.error("Error getting topic info: {}".format(e))


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python topic_modeling_pipeline.py <partition_models>")
        sys.exit(1)

    partition_models = sys.argv[1]
    pipeline = TopicModelingPipeline(partition_models = partition_models)
    pipeline.run_pipeline()
