# from modules.recommender import Recommender
# import pandas as pd
# import numpy as np

# # Test input movie name
# movies_name = "horror"  # Replace with an actual movie name from your dataset

# # Initialize the Recommender class
# recommender = Recommender()

# # Call the recommend function
# recommendations = recommender.recommend(movies_name)

# # Print the results
# print("Recommendations:", recommendations)


# # Load graph
# with open("models/graph_model.pkl", "rb") as f:
#     G = pickle.load(f)

# # Print some nodes
# print("Sample nodes:", list(G.nodes)[:20])

# # Print sample edges
# print("Sample edges:", list(G.edges)[:20])
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging
import torch
from fastapi import FastAPI
import uvicorn

# Function to get the version of a module
def get_version(module):
    return getattr(module, "__version__", "Version info not available")

# Dictionary to store the versions
versions = {
    "numpy": get_version(np),
    "pandas": get_version(pd),
    "networkx": get_version(nx),
    "faiss": get_version(faiss),
    "sentence_transformers": get_version(SentenceTransformer),
    "torch": get_version(torch),
    "fastapi": get_version(FastAPI),
    "uvicorn": get_version(uvicorn)
}

# Print the versions
for lib, version in versions.items():
    print(f"{lib}: {version}")
