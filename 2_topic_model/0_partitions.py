import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import os
import logging
import sys
import pyarrow.parquet as pq
import pyarrow as pa

sys.setrecursionlimit(10**6)

class EmbeddingPipeline:
    def __init__(self):

        ##load the data--------------------------
        self.data_path = "../../data_update_2025/update_text_table.parquet"
        
        self.df = None
        self.texts = None

        ##save the embeddings-------------------------- 
        self.log_folder = "../../job_outputs/partitions_update"

        # Ensure log folder exists
        os.makedirs(self.log_folder, exist_ok=True)

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
        log_file = os.path.join(self.log_folder, 'pipeline.log')  # Define log file path
        fh = logging.FileHandler(log_file)  # File handler for logging to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)  # Add file handler to logger

    def load_data(self):
        try:
            self.logger.info("Loading data from: {}".format(self.data_path))
            self.df = pd.read_parquet(self.data_path)
        except Exception as e:
            self.logger.error("Error loading data: {}".format(e))

    def save_partitions(self):
        try:
            self.logger.info("Starting to save partitions")
            num_rows = len(self.df)
            partition_size = 500000
            num_partitions = (num_rows + partition_size - 1) // partition_size  # Calculate number of partitions

            for i in range(num_partitions):
                start_row = i * partition_size
                end_row = min((i + 1) * partition_size, num_rows)
                partition_df = self.df.iloc[start_row:end_row]
                partition_file = os.path.join(self.log_folder, f'partition_{i + 1}.parquet')
                partition_df.to_parquet(partition_file)
                self.logger.info(f"Saved partition {i + 1} to {partition_file}")

            self.logger.info("All partitions saved successfully.")
        except Exception as e:
            self.logger.error("Error saving partitions: {}".format(e))

    def run_pipeline(self):
        try:
            self.logger.info("Starting pipeline")
            self.load_data()
            self.save_partitions()
            self.logger.info("Pipeline completed successfully.")
        except Exception as e:
            self.logger.error("Pipeline failed with error: {}".format(e))


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 1:
        print("Usage: python partition_pipeline.py")
        sys.exit(1)

    pipeline = EmbeddingPipeline()
    pipeline.run_pipeline()
