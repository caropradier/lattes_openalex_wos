import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import os
import logging
import sys
from datetime import datetime
from threadpoolctl import threadpool_limits
from nltk.corpus import stopwords
import pyarrow.parquet as pq
import pyarrow as pa

sys.setrecursionlimit(10**6)

class TopicModelingPipeline:
    def __init__(self, partition, min_cluster_size=100):

        self.partition = str(partition)

        ####Documents
        self.data_path = f"../../job_outputs/partitions_update/partition_{self.partition}.parquet"

        ####Embeddings
        self.embeddings_path = f"../../job_outputs/embedding_update/embedding_{self.partition}.parquet"

        self.min_cluster_size = int(min_cluster_size)

        ####Topic Model
        self.log_folder = "../../job_outputs/topic_model_update"
        

        # Ensure log folder exists
        os.makedirs(self.log_folder, exist_ok=True)

        self.df = None
        self.embeddings = None
        self.texts = None
        self.disciplines = None
        self.pca_embeddings = None
        self.umap_model = None
        self.vectorizer_model = None
        self.hdbscan_model = None
        self.topic_model = None

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
        log_file = os.path.join(self.log_folder, "pipeline.log")  # Define log file path
        fh = logging.FileHandler(log_file)  # File handler for logging to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)  # Add file handler to logger

    def load_data(self):
        try:
            self.logger.info("Loading data from: {}".format(self.data_path))
            self.df = pd.read_parquet(self.data_path)
            self.texts = self.df['Titulo'].values
            self.logger.info("Data loaded successfully. Number of texts: {}".format(len(self.texts)))
        except Exception as e:
            self.logger.error("Error loading data: {}".format(e))

    def load_embeddings(self):
        """
        Load embeddings from a parquet file specified by `self.embeddings_path`.
        This method attempts to read embeddings from a parquet file into a pandas DataFrame,
        converts the DataFrame to a NumPy array, and assigns it to `self.embeddings`.
        It also logs the process, including the shape of the loaded embeddings.
        Raises:
            Exception: If there is an error during the loading process, it logs the error message.
        """
        try:
            self.logger.info("Loading embeddings from: {}".format(self.embeddings_path))
            embeddings_df = pd.read_parquet(self.embeddings_path)
        
            # Convert DataFrame to NumPy array
            self.embeddings = embeddings_df.values
        
            # Ensure the shape is as expected
            self.logger.info("Embeddings loaded successfully. Shape: {}".format(self.embeddings.shape))
        except Exception as e:
            self.logger.error("Error loading embeddings: {}".format(e))

    def preprocess_embeddings(self):
        """
        Preprocesses the embeddings by applying PCA for dimensionality reduction and rescaling the result.

        This method performs the following steps:
        1. Logs the start of the preprocessing.
        2. Limits the number of threads to 1 for the PCA computation.
        3. Applies PCA to reduce the embeddings to 5 components.
        4. Rescales the PCA-transformed embeddings.
        5. Logs any errors encountered during the process.

        Attributes:
            self.embeddings (numpy.ndarray): The original embeddings to be processed.
            self.pca_embeddings (numpy.ndarray): The PCA-transformed and rescaled embeddings.
            self.logger (logging.Logger): Logger for logging information and errors.

        Raises:
            Exception: If an error occurs during the preprocessing of embeddings.
        """
        try:
            self.logger.info("Preprocessing embeddings")
            with threadpool_limits(limits=1):
                self.pca_embeddings = self.rescale(PCA(n_components=5).fit_transform(self.embeddings))
        except Exception as e:
            self.logger.error("Error preprocessing embeddings: {}".format(e))

    def rescale(self, x, inplace=False):
        """
        Rescales the input array `x` by dividing it by the standard deviation of its first column multiplied by 10000.

        Parameters:
        x (array-like): The input array to be rescaled.
        inplace (bool, optional): If True, the rescaling is done in-place. If False, a copy of the array is rescaled. Default is False.

        Returns:
        array-like: The rescaled array.

        Raises:
        Exception: If an error occurs during rescaling, it logs the error message.
        """
        try:
            self.logger.info("Rescaling embeddings")
            if not inplace:
                x = np.array(x, copy=True)
            x /= np.std(x[:, 0]) * 10000
            return x
        except Exception as e:
            self.logger.error("Error rescaling embeddings: {}".format(e))

    def initialize_umap_model(self):
        """
        Initializes the UMAP model with predefined parameters.

        This method sets up the UMAP (Uniform Manifold Approximation and Projection) model 
        with specific parameters such as number of neighbors, number of components, minimum 
        distance, metric, initial embeddings, random state, number of jobs, and memory usage.

        Raises:
        Exception: If there is an error during the initialization of the UMAP model, 
                   it logs the error message.
        """
        try:
            self.logger.info("Initializing UMAP model")
            self.umap_model = UMAP(
                n_neighbors=10,
                n_components=5,
                min_dist=0.01,
                metric="cosine",
                init=self.pca_embeddings,
                random_state=1234,
                n_jobs=10,
                low_memory=True
            )
        except Exception as e:
            self.logger.error("Error initializing UMAP model: {}".format(e))

    def initialize_vectorizer_model(self):
        """
        Initializes the CountVectorizer model with custom stopwords and specified parameters.

        This method sets up the CountVectorizer model for text processing. It includes a custom list of stopwords
        to be excluded during the vectorization process. The vectorizer is configured to consider unigrams and bigrams,
        and it filters terms based on document frequency thresholds.

        Raises:
        Exception: If there is an error during the initialization of the CountVectorizer model, it logs the error message.
        """
        try:
            self.logger.info("Initializing CountVectorizer model")
            custom_stopwords = (stopwords.words(stopwords.fileids()) + ['ttttusepackage', 'amssymb', 'amsmath', 'mathrsfs', 'amsbsy',
                   'amsfonts', 'oddsidemargin', 'wasysym', '69pt', 'upgreek', '12pt', 'documentclass',
                   'ttttbegin', 'ttttsetlength', 'document', 'mso', 'font', 'tstyle', '4pt', '0pt',
                   'style', '0cm', 'margin', 'calibri', '0001pt', 'qformat', 'msonormaltable', 'pagination', 'rowband', 'size',
                   '0400', 'colband', 'bottom'] + ['netcommons','retraction', 'retracted', 'notice retraction', 'iop publishing', 'elsevier', 'iop', 'withdrawn', 'article withdrawal', 
                   'conference', 'proceedings', 'international conference', 'welcome', 'conference proceedings', 'committee', 'workshop', 'proceedings record',
                   'ieee', 'ieee transactions', 'listing', 'committee members', 'members society', 'society officers', 'committee', 'listing current', 'publication', 'information authors',
                   'editorial', 'reviewers', 'editor', 'number submissions', 'submissions', 'editors', 'publication', 'reviewer',
                   'resumen', 'palabras clave', 'introducción', 'método', 'resultados', 'discusión', 'conclusión', 'agradecimientos', 'referencias', 'abstract', 'keywords', 'introduction', 'method', 'results', 'discussion', 'conclusion', 'acknowledgements', 'references',
                   'corrects article', 'article 10', 'corrects', 'erratum', 'corrigendum', 'paper published', 'published accessed', 'author correction', 'amendment paper', 'version article',
                   'pp', 'isbn', 'press', 'book', 'university press', 'book reviews', 'london', 'cambridge', 'book review', 'edited',
                   'advertisement', 'ieee', 'advertisement ieee', 'computer society', 'ieee computer', 'advertisement advertisement', 'org', 'www ieee', 'ieee org', 'cacm',
                   'collabratec', 'ieee collabratec', 'ieee', 'interests', 'introducing ieee', 'learn ieee', 'content featuring', 'contributing knowledgeable', 'connect geographic', 'network collaborate',
                   'reply gastroenterology','gastroenterology 2002', 'gastroenterology 1998', 'gastroenterology 2001',
                   'resumen', 'revista', 'netcommons', 'cms netcommons', 'netcommons maple', 'isbn', 'maple', 'resúmenes', 'reseña',
                   'artigo', 'resumo', 'partir', 'palavras', 'trabalho', 'estudo', 'análise', 'resultados', 'conclusões', 'referências',  
                   'online public', 'access catalog', 'public access', 'catalog', 'online', 'northern research', 'station usda', 'usda forest', 'netcommons', 'cms netcommons',
                   'presenta resumen', 'resumen', 'presenta', 'artículo presenta', 'reseña', 'reseña presenta', 'economía', 'artículo', 'revista',
                   'artikkelen','artiklen',  'journal',
                   'cacm', 'acm org', 'cacm acm', 'blog cacm', 'advertisement', 'http cacm', 'http', 'acm', 'blog',
                   'related content', '2017 related', 'march april', 'april 2017', 'reviewer', 'reviewer acknowledgements',  'acknowledgements', 'editorial', 'vol',
                   'cet','cette','article','cet article', 
                    'journal','journal curriculum', 
                   'http dx', 'dx org', 'org 10', 'dx', 'org', 'http', '10 3329', '3329', '10 4038', '4038',
                   'br dermatol', 'linked article', 'dermatol 2021', 'dermatol', '2021 185', 'br', '185', '2021 184', '我们发现', 'cadi',
                   '2016 abstract', 'week articles', 'articles', 'papers note', 'note science', 'week', 'papers', 'note', 'articles describe', 'sci transl',
                   'scipost', 'submission detail', 'scipost submission', 'submission', 'scipost journals', 'detail scipost', 'publication detail', 'journals publication', 'scipost phys', 'journals',
                   "Foreword Japan's largest platform for academic e-journals: J-STAGE is a full text database for reviewed academic papers published by Japanese societies",
                   'platform academic', 'journals stage', 
                   'present summary', 'article present', 'presenta resumen', 'resumen', 'presenta', 'contiene', 'review present', 'contiene resumen', 'artículo presenta', 'resúmen',
                   'information authors', 'authors instructions', 'instructions', 'publishing journal', 'authors publishing', 'instructions give', 'publishing', 'guidelines preparing', 'preparing papers', 'papers publication',
                   'Sources: Encyclopedia of Evolution Sources: Encyclopedia of Evolution',
                   'call papers', 'submit unpublished', 'inclusion upcoming', 'requested submit', 'described call', 'authors requested', 'manuscripts inclusion', 'upcoming event', 'unpublished manuscripts', 'event described',
                   'reply', 'reply reply', 'reply comment', 'comment', 'replies reply', 'replies', 'authors comment',
                   'attached pdf', 'russian version', 'file english', 'article attached', 'version russian', 'pdf file', 'full text', 'english version', 'english full', 'text article',
                   'chez', 'cas', 'étude', 'traitement', 'cette', 'résultats', 'travail', 'être', 'prise', 'après',
                   'Microscopy and Microanalysis','complete issue', 'issue complete',
                   'cover image', 'cover', 'image', '10 1002', '1002', 'image volume', 'issue cover', 'cover picture', 'journal', '2017 cover',
                   'ressenya', 'paraules', 'clau', 'paraules clau', '978 84', 'resum', 'estudi', 
                   'resúmenes palabras', 'número incluyeron', 'incluyeron resúmenes', 'resúmenes', 'incluyeron', 'palabras clave', 'número', 'clave', 'palabras',
                   'present summary', 'article present', 'presenta resumen', 'resumen', 'presenta', 'contiene', 'review present', 'contiene resumen', 'artículo presenta', 'resúmen',
                   'zusammenfassung'])
            self.vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords,
                                                    max_df=0.8, min_df=0.001)
        except Exception as e:
            self.logger.error("Error initializing CountVectorizer model: {}".format(e))

    def initialize_hdbscan_model(self):
        """
        Initializes the HDBSCAN model with the specified parameters.

        This method sets up the HDBSCAN model using the parameters defined in the class attributes.
        It logs the initialization process and handles any exceptions that may occur during the setup.

        Raises:
        Exception: If there is an error during the initialization of the HDBSCAN model, it logs the error message.
        """
        try:
            self.logger.info("Initializing HDBSCAN model")
            self.hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=10, metric='euclidean',
                                         cluster_selection_epsilon=0.05, cluster_selection_method='leaf', prediction_data=True,
                                         core_dist_n_jobs=16, memory='tmp/')
        except Exception as e:
            self.logger.error("Error initializing HDBSCAN model: {}".format(e))
    
    def initialize_KeyBERTInspired(self):
        """
        Initializes the KeyBERTInspired model and assigns it to the representation_model attribute.
        
        This method logs the initialization process and handles any exceptions that may occur during 
        the initialization. If an error occurs, it logs the error message.
        
        Raises:
            Exception: If there is an error during the initialization of the KeyBERTInspired model.
        """
        try:
            self.logger.info("Initializing KeyBERTInspired model")
            self.representation_model = KeyBERTInspired()
        except Exception as e:
            self.logger.error("Error Initializing KeyBERTInspired model: {}".format(e))

    def fit_topic_model(self):
        """
        Fits a semi-supervised BERTopic model to the provided texts and embeddings.
        This method initializes and fits a BERTopic model using the provided parameters and data. 
        It also handles outliers by reducing them and updating the topics accordingly.
        Attributes:
            self.logger (Logger): Logger instance for logging information and errors.
            self.topic_model (BERTopic): Instance of the BERTopic model.
            self.min_cluster_size (int): Minimum size of the clusters.
            self.sentence_model (Any): Pre-trained sentence embedding model.
            self.vectorizer_model (Any): Vectorizer model for text vectorization.
            self.hdbscan_model (Any): HDBSCAN model for clustering.
            self.umap_model (Any): UMAP model for dimensionality reduction.
            self.representation_model (Any): Model for topic representation.
            self.texts (List[str]): List of texts to be used for topic modeling.
            self.embeddings (List[Any]): List of embeddings corresponding to the texts.
            self.disciplines (List[Any]): List of discipline labels for semi-supervised learning.
        Raises:
            Exception: If an error occurs during the fitting process, it is logged and raised.
        """
        try:
            self.logger.info("Fitting semi-supervised BERTopic model")
            self.topic_model = BERTopic(verbose=True, min_topic_size=self.min_cluster_size,
                                        embedding_model=self.sentence_model, low_memory=True, calculate_probabilities=False,
                                        vectorizer_model=self.vectorizer_model, hdbscan_model=self.hdbscan_model,
                                        umap_model=self.umap_model, representation_model=self.representation_model)
            topics, probabilities = self.topic_model.fit_transform(self.texts, self.embeddings)
            
            new_topics = self.topic_model.reduce_outliers(self.texts, topics, strategy="embeddings", threshold=0.1)
            self.topic_model.update_topics(self.texts, topics=new_topics, vectorizer_model=self.vectorizer_model)
        except Exception as e:
            self.logger.error("Error fitting BERTopic model: {}".format(e))

    def save_topic_model(self, save_path):
        """
        Save the BERTopic model to the specified path.

        Parameters:
        save_path (str): The file path where the BERTopic model will be saved.

        This method attempts to save the BERTopic model using the specified path.
        It logs the process and handles any exceptions that may occur during saving.

        Raises:
        Exception: If there is an error during the saving process, it will be logged.
        """
        try:
            self.logger.info("Saving BERTopic model to: {}".format(save_path))
            self.topic_model.save(save_path, serialization='safetensors', save_ctfidf=True, save_embedding_model=self.sentence_model)
        except Exception as e:
            self.logger.error("Error saving BERTopic model: {}".format(e))
    
    def save_topic_info(self, file_path):
        """
        Save the topic information to a CSV file.

        Parameters:
        file_path (str): The path to the file where the topic information will be saved.

        Raises:
        Exception: If there is an error during the saving process, it will be logged.
        """
        try:
            self.logger.info("Saving topic info to file: {}".format(file_path))
            topic_info = self.get_topic_info()
            topic_info.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error("Error saving topic info: {}".format(e))

    def run_pipeline(self):
        try:
            model_save_path = os.path.join(self.log_folder, f"topic_model_{self.partition}")
            topic_info_file = os.path.join(self.log_folder, f"topic_info_{self.partition}.csv")

            self.logger.info("Starting pipeline")
            self.load_data()
            #self.sentence_model = SentenceTransformer("jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br")
            self.sentence_model = SentenceTransformer("../../../models/jvanhoof_model")
            self.load_embeddings()
            self.preprocess_embeddings()
            self.initialize_vectorizer_model()
            self.initialize_hdbscan_model()
            self.initialize_umap_model()
            self.initialize_KeyBERTInspired()
            self.fit_topic_model()
            self.save_topic_model(model_save_path)
            self.save_topic_info(topic_info_file)
            self.logger.info("Pipeline completed successfully. Model saved in: {}".format(model_save_path))
        except Exception as e:
            self.logger.error("Error in pipeline execution: {}".format(e))

    def get_topic_info(self):
        try:
            self.logger.info("Getting topic info")
            return self.topic_model.get_topic_info()
        except Exception as e:
            self.logger.error("Error getting topic info: {}".format(e))

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python topic_modeling_pipeline.py <partition>")
        sys.exit(1)

    partition = sys.argv[1]
    pipeline = TopicModelingPipeline(partition = partition)
    pipeline.run_pipeline()