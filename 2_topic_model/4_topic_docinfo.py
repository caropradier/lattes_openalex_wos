import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
import logging
import sys
import pyarrow.parquet as pq
import pyarrow as pa
import gc

sys.setrecursionlimit(10**6)

class TopicModelingPipeline:
    def __init__(self, data_path_existing, model_folder):
        self.data_path_existing = data_path_existing
        self.model_folder = model_folder
        self.log_folder = model_folder
        self.df_existing = None
        self.df = None
        self.texts = None
        self.restored_model = None

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

    def load_data(self):
        try:
            self.logger.info(f"Loading data from: {self.data_path_existing}")
        
            # Load existing data from multiple partitions
            try:
                self.df = pd.DataFrame()
                for i in range(1, 11):
                    file_path = os.path.join(self.data_path_existing, f"partition_{i}.parquet")
                    self.logger.info(f"Loading partition: {file_path}")
                    partition_df = pd.read_parquet(file_path)
                    self.df = pd.concat([self.df, partition_df], ignore_index=True)

                self.logger.info("All partitions loaded and concatenated successfully")
            except Exception as e:
                self.logger.error(f"Error loading existing data: {e}")

            self.texts = self.df['Titulo'].values
            self.logger.info(f"Data loaded successfully. Number of texts: {len(self.texts)}")
    
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")

    def load_model(self):
        try:
            self.logger.info("Loading model from: {}".format(self.model_folder))
            self.restored_model = BERTopic.load(self.model_folder,embedding_model=self.sentence_model)
            self.logger.info("Existing model loaded successfully")
        except Exception as e:
            self.logger.error("Error loading model: {}".format(e))


    def save_document_topics_in_chunks(self, file_path, chunk_size=5000000):
        try:
            self.logger.info("Saving document topics to file in chunks: {}".format(file_path))
            document_info = self.get_document_info()

            # Combine the necessary columns
            doc_topic_df = pd.concat([self.df[['id']], self.df[['Titulo']], document_info[['Topic']]], axis=1)
            doc_topic_df = doc_topic_df.drop_duplicates()
            writer = None

            for start_row in range(0, len(doc_topic_df), chunk_size):
                end_row = start_row + chunk_size
                chunk = doc_topic_df.iloc[start_row:end_row]
                table = pa.Table.from_pandas(chunk)

                if writer is None:
                    writer = pq.ParquetWriter(file_path, table.schema)

                writer.write_table(table)
                self.logger.info(f"Saved document topics chunk rows {start_row} to {end_row} to {file_path}")

            if writer is not None:
                writer.close()
                self.logger.info(f"Finished saving all document topics chunks to {file_path}")
        
        except Exception as e:
            self.logger.error("Error saving document topics in chunks: {}".format(e))
            raise


    def run_pipeline(self):
        try:
            model_save_path = os.path.join(self.log_folder)
            document_topics_file = os.path.join(self.log_folder, f"merged_document_topics.parquet")

            self.logger.info("Starting pipeline")
            self.load_data()
            self.sentence_model = SentenceTransformer("jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br")
            self.load_model()
            self.save_document_topics_in_chunks(file_path=document_topics_file)
            self.logger.info("Pipeline completed successfully. Model saved in: {}".format(model_save_path))
        except Exception as e:
            self.logger.error("Error in pipeline execution: {}".format(e))

    def get_document_info(self):
        try:
            self.logger.info("Getting document info")
            return self.restored_model.get_document_info(self.texts)
        except Exception as e:
            self.logger.error("Error getting document info: {}".format(e))

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        print("Usage: python 4_topic_docinfo.py <data_path_existing> <model_folder>")
        sys.exit(1)

    data_path_existing = sys.argv[1]
    model_folder = sys.argv[2]
    pipeline = TopicModelingPipeline(data_path_existing = data_path_existing, model_folder = model_folder)
    pipeline.run_pipeline()
