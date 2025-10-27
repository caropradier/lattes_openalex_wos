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
    def __init__(self,partition):

        self.partition = str(partition)

        ##load the data--------------------------
        self.data_path = f"../../job_outputs/partitions_update/partition_{self.partition}.parquet"
        
        self.embedding_file = f"embedding_{self.partition}.parquet"
        self.df = None
        self.texts = None

        ##save the embeddings-------------------------- 
        self.log_folder = "../../job_outputs/embedding_update"

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
            self.texts = self.df['Titulo'].values
        except Exception as e:
            self.logger.error("Error loading data: {}".format(e))

    def encode_and_save_embeddings(self):
        """
        Encodes the texts into embeddings using a SentenceTransformer model and saves the embeddings.

        This method initializes a SentenceTransformer model with the "intfloat/multilingual-e5-large-instruct" model,
        encodes the texts into embeddings using multi-process pooling, and saves the embeddings. If an error occurs
        during the process, it logs the error message.

        Raises:
            Exception: If an error occurs during encoding or saving embeddings, it logs the error message.
        """
        try:
            self.logger.info("Encoding and saving embeddings")
            sentence_model = SentenceTransformer("jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br")
            pool = sentence_model.start_multi_process_pool(target_devices=['cuda:0', 'cuda:1'])
            embeddings = sentence_model.encode_multi_process(self.texts, pool=pool)
            sentence_model.stop_multi_process_pool(pool)
            self.sentence_model = sentence_model
            self.save_embeddings(embeddings)
        except Exception as e:
            self.logger.error("Error encoding and saving embeddings: {}".format(e))

    def save_embeddings(self, embeddings):
        try:
            self.logger.info("Saving embeddings in Parquet format")
            embeddings_df = pd.DataFrame(embeddings)
            table = pa.Table.from_pandas(embeddings_df)
            pq.write_table(table, os.path.join(self.log_folder, self.embedding_file))
        except Exception as e:
            self.logger.error("Error saving embeddings: {}".format(e))

    def run_pipeline(self):
        try:
            self.logger.info("Starting pipeline")
            self.load_data()
            self.encode_and_save_embeddings()
            self.logger.info("Pipeline completed successfully.")
        except Exception as e:
            self.logger.error("Pipeline failed with error: {}".format(e))


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python embedding_pipeline.py <partition>")
        sys.exit(1)

    partition = sys.argv[1]
    pipeline = EmbeddingPipeline(partition = partition)
    pipeline.run_pipeline()
