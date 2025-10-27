import os
import pandas as pd
import openai
import logging

# Note: before running add the api key to the terminal session using:
# export OPENAI_API_KEY="your-openai-api-key"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("topic_labeler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TopicLabeler:
    def __init__(self, data_path, output_path, openai_api_key=None):
        self.data_path = data_path
        self.output_path = output_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        #self.client = openai.OpenAI(api_key=self.openai_api_key)

        logger.info("TopicLabeler initialized with data path: %s and output path: %s", 
                    self.data_path, self.output_path)

    def load_data(self):
        """Load the CSV file containing the topic model results."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info("Data loaded from %s. Shape: %s", self.data_path, self.df.shape)
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            raise

    def generate_labels(self):
        """Generate short labels for each topic using OpenAI's GPT."""
        try:
            self.df['Label'] = self.df.apply(self._generate_label_for_topic, axis=1)
            logger.info("Labels generated for all topics.")
        except Exception as e:
            logger.error("Failed to generate labels: %s", e)
            raise

    def _generate_label_for_topic(self, row):
        """Generate a label for a single topic."""
        prompt = f"Create a short and accurate label in English (2-3 words) for the following topic based on its representation:\n\nTopic: {row['Topic']}\nRepresentation: {row['Representation']}\n\nLabel (your output should only include the label):"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5
        )
        label = response.choices[0].message['content'].strip()
        logger.info("Generated label for topic '%s': %s", row['Topic'], label)
        return label

    def save_results(self):
        """Save the labeled topics to a CSV file."""
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info("Results saved to %s", self.output_path)
        except Exception as e:
            logger.error("Failed to save results: %s", e)
            raise

# Example usage
if __name__ == "__main__":
    data_path = "job_outputs/merged_model_update/merged_topic_info.csv"
    output_path = "job_outputs/merged_model_update/labeled_topic_info.csv"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    labeler = TopicLabeler(data_path, output_path, openai_api_key)
    labeler.load_data()
    labeler.generate_labels()
    labeler.save_results()
