# import networkx as nx
# import faiss
# import numpy as np
# import pandas as pd
# import pickle
# from sentence_transformers import SentenceTransformer
# import logging
# import os

# # Setup logging
# logging.basicConfig(filename="logs/train.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def train_model():
#     try:
#         # Load preprocessed data
#         df = pd.read_csv("data/processed_data.csv")

#         # Create Graph
#         G = nx.Graph()
#         for user_id in df["userId"].unique():
#             G.add_node(f"U{user_id}", type="user")

#         for movie_id, title in zip(df["movieId"], df["title"]):
#             G.add_node(f"M{movie_id}", title=title, type="movie")

#         for _, row in df.iterrows():
#             G.add_edge(f"U{row['userId']}", f"M{row['movieId']}", weight=row["rating"])

#         # Save Graph Model
#         os.makedirs("models", exist_ok=True)
#         with open("models/graph_model.pkl", "wb") as f:
#             pickle.dump(G, f)
#         logging.info("Graph model saved successfully.")

#         # Train FAISS Model
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         movie_texts = df["title"] + " " + df["genres"]
#         movie_vectors = model.encode(movie_texts.tolist())

#         # Save FAISS Index
#         index = faiss.IndexFlatL2(movie_vectors.shape[1])
#         index.add(np.array(movie_vectors))
#         faiss.write_index(index, "models/faiss_index.bin")
#         logging.info("FAISS index saved successfully.")

#     except Exception as e:
#         logging.error(f"Error during training: {e}")
#         raise e

# if __name__ == "__main__":
#     train_model()





import networkx as nx
import faiss
import numpy as np
import pandas as pd
import pickle
import torch
from sentence_transformers import SentenceTransformer
import os
import logging

# Setup logging
logging.basicConfig(filename="logs/train.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the improved embedding model (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

def train_model():
    try:
        # Load preprocessed data
        df = pd.read_csv("data/processed_data.csv")
        print("-----------------------STARTED--------------------------")
        # Remove duplicate movies by averaging ratings
        df_unique = df.groupby(["movieId", "title", "genres"], as_index=False).mean()

        # Create Graph
        G = nx.Graph()
        for user_id in df["userId"].unique():
            G.add_node(f"U{user_id}", type="user")

        for _, row in df_unique.iterrows():
            G.add_node(f"M{row['movieId']}", title=row["title"], type="movie")

        for _, row in df.iterrows():
            G.add_edge(f"U{row['userId']}", f"M{row['movieId']}", weight=row["rating"])

        # Save Graph Model
        os.makedirs("models", exist_ok=True)
        with open("models/graph_model.pkl", "wb") as f:
            pickle.dump(G, f)
        logging.info("Graph model saved successfully.")

        # Generate better embeddings
        movie_texts = df_unique["title"] + " Genre: " + df_unique["genres"]
        movie_vectors = model.encode(movie_texts.tolist(), normalize_embeddings=True)

        # Use FAISS IndexHNSWFlat for speed
        d = movie_vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)  # Faster than IndexFlatL2
        index.add(np.array(movie_vectors))
        faiss.write_index(index, "models/faiss_index.bin")
        logging.info("FAISS index saved successfully.")
        print("----------------DONE------------------------------")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e

if __name__ == "__main__":
    train_model()
