import pickle
import faiss
import networkx as nx
import pandas as pd
import numpy as np
import os
import logging
import torch
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(filename="logs/update_model.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load embedding model (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# Paths
GRAPH_MODEL_PATH = "models/graph_model.pkl"
FAISS_INDEX_PATH = "models/faiss_index.bin"
NEW_DATA_PATH = "data/new_movies.csv"


def load_existing_models():
    """Load existing graph and FAISS index if available."""
    try:
        if os.path.exists(GRAPH_MODEL_PATH):
            with open(GRAPH_MODEL_PATH, "rb") as f:
                G = pickle.load(f)
        else:
            G = nx.Graph()
            logging.warning("No existing graph model found. Creating a new one.")

        if os.path.exists(FAISS_INDEX_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            index = None
            logging.warning("No existing FAISS index found. A new one will be created.")

        return G, index
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return nx.Graph(), None


def update_graph(G, df_new):
    """Update the graph with new users, movies, and ratings."""
    try:
        existing_users = set(n for n, d in G.nodes(data=True) if d.get("type") == "user")
        existing_movies = set(n for n, d in G.nodes(data=True) if d.get("type") == "movie")

        for _, row in df_new.iterrows():
            user_node = f"U{row['userId']}"
            movie_node = f"M{row['movieId']}"

            if user_node not in existing_users:
                G.add_node(user_node, type="user")
                existing_users.add(user_node)

            if movie_node not in existing_movies:
                G.add_node(movie_node, title=row["title"], type="movie")
                existing_movies.add(movie_node)

            if G.has_edge(user_node, movie_node):
                G[user_node][movie_node]["weight"] = (G[user_node][movie_node]["weight"] + row["rating"]) / 2  # Average rating
            else:
                G.add_edge(user_node, movie_node, weight=row["rating"])

        # Save updated graph
        with open(GRAPH_MODEL_PATH, "wb") as f:
            pickle.dump(G, f)

        logging.info(f"Graph updated with {len(df_new)} new interactions.")
    except Exception as e:
        logging.error(f"Error updating graph: {e}")


def update_faiss_index(G, df_new, index):
    """Compute embeddings for new movies and update FAISS index."""
    try:
        existing_movies = set(n for n, d in G.nodes(data=True) if d.get("type") == "movie")
        df_new_movies = df_new[~df_new["movieId"].astype(str).apply(lambda x: f"M{x}" in existing_movies)]

        if df_new_movies.empty:
            logging.info("No new movies found for FAISS update.")
            return index

        # Compute embeddings
        movie_texts = df_new_movies["title"] + " Genre: " + df_new_movies["genres"]
        new_movie_vectors = model.encode(movie_texts.tolist(), normalize_embeddings=True)

        # Initialize FAISS if not available
        d = new_movie_vectors.shape[1]  # Embedding dimension
        if index is None:
            index = faiss.IndexFlatL2(d)

        index.add(np.array(new_movie_vectors))  # Add new vectors
        faiss.write_index(index, FAISS_INDEX_PATH)  # Save updated index

        logging.info(f"Added {len(df_new_movies)} new movies to FAISS index.")
        return index
    except Exception as e:
        logging.error(f"Error updating FAISS index: {e}")
        return index


def main():
    """Main function to update the model with new data."""
    try:
        # Load existing models
        G, index = load_existing_models()
        print("--------------UPDATED MODEL----------------------------")
        # Load new data
        if not os.path.exists(NEW_DATA_PATH):
            logging.error(f"New data file {NEW_DATA_PATH} not found.")
            return

        df_new = pd.read_csv(NEW_DATA_PATH)
        if df_new.empty:
            logging.info("New data file is empty. No updates needed.")
            return

        # Update graph and FAISS index
        update_graph(G, df_new)
        index = update_faiss_index(G, df_new, index)

        logging.info("Model update completed successfully!")
    except Exception as e:
        logging.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
