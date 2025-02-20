import pandas as pd
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/preprocess.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data():
    try:
        # Load datasets
        movies = pd.read_csv("data/movies.csv")
        ratings = pd.read_csv("data/ratings.csv")

        # Merge datasets
        df = ratings.merge(movies, on="movieId").drop(columns=["timestamp"])

        # Save preprocessed data
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/processed_data.csv", index=False)

        logging.info("Data preprocessing successful. Saved to data/processed_data.csv")
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise e  # Raise for debugging

if __name__ == "__main__":
    preprocess_data()
