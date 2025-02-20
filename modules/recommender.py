import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
import os

# Setup logging
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# class Recommender:
#     def __init__(self):
#         try:
#             # Load models
#             with open("models/graph_model.pkl", "rb") as f:
#                 self.G = pickle.load(f)

#             self.index = faiss.read_index("models/faiss_index.bin")
#             self.model = SentenceTransformer("all-MiniLM-L6-v2")

#             # Load movie data
#             self.df = pd.read_csv("data/processed_data.csv")
#             logging.info("Models and data loaded successfully.")

#         except Exception as e:
#             logging.error(f"Error loading models: {e}")
#             raise e

#     def recommend(self, query):
#         try:
#             query_vector = self.model.encode([query])
#             _, indices = self.index.search(np.array([query_vector]), k=5)
#             recommended_movies = self.df.iloc[indices[0]]["title"].tolist()

#             logging.info(f"User Query: {query} | Recommendations: {recommended_movies}")
#             return recommended_movies
        
#         except Exception as e:
#             logging.error(f"Error during recommendation: {e}")
#             return ["Error generating recommendations."]
        

# import numpy as np
# import logging

import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Recommender:
    def __init__(self):
        try:
            # Load models
            with open("models/graph_model.pkl", "rb") as f:
                self.G = pickle.load(f)

            self.index = faiss.read_index("models/faiss_index.bin")
            # self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

            # Load movie data
            self.df = pd.read_csv("data/processed_data.csv")

            # Debugging: Check FAISS index dimension
            logging.info(f"FAISS index dimension: {self.index.d}")
            logging.info(f"SentenceTransformer embedding dimension: {self.model.get_sentence_embedding_dimension()}")

            logging.info("Models and data loaded successfully.")

        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise e

    def recommend(self, query):
        try:
            # Generate embedding for query
            query_vector = self.model.encode([query])
            
            # Ensure query_vector is a 2D numpy array (1, d)
            query_vector = np.array(query_vector).astype("float32")
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            # Check FAISS index and query vector dimension
            if query_vector.shape[1] != self.index.d:
                logging.error(f"Dimension mismatch: Query vector ({query_vector.shape[1]}) vs FAISS index ({self.index.d})")
                return ["Error: Query vector dimension mismatch"]

            # FAISS search
            search_result = self.index.search(query_vector, k=5)

            # Debugging: Log FAISS output
            logging.info(f"FAISS search result: {search_result}")

            # Ensure FAISS search output has exactly two values
            if not isinstance(search_result, tuple) or len(search_result) != 2:
                logging.error(f"Unexpected FAISS output format: {search_result}")
                return ["Error generating recommendations."]

            distances, indices = search_result  # Proper unpacking

            # Ensure valid recommendations
            if indices is None or len(indices[0]) == 0:
                logging.error("FAISS returned no valid indices.")
                return ["No recommendations found."]

            recommended_movies = self.df.iloc[indices[0]]["title"].tolist()
            logging.info(f"User Query: {query} | Recommendations: {recommended_movies}")
            return recommended_movies

        except Exception as e:
            logging.error(f"Error during recommendation: {e}")
            return ["Error generating recommendations."]
