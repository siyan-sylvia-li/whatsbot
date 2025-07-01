import faiss
import openai
import numpy as np
from typing import List, Dict, Any, Tuple
import dspy
import os
import json

# ----------------------
# Configuration
# ----------------------

embedder = dspy.Embedder(model="openai/text-embedding-3-small", caching=True)


# ----------------------
# Embedding Utility
# ----------------------
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI.
    Returns an array of shape (len(texts), embedding_dim).
    """
    return embedder(inputs=texts)

# ----------------------
# FAISS Index with Metadata
# ----------------------
class FaissIndexWithMetadata:
    def __init__(self, user_prefix: str, dimension: int = 0):
        """
        Initialize a FAISS index and storage for metadata.
        """
        self.dimension = dimension
        index_file = user_prefix + ".faiss"
        metadata_file = user_prefix + ".json"
        self.index_file = index_file
        self.metadata_file = metadata_file
        if not os.path.exists(index_file):
            self.index = faiss.IndexFlatIP(dimension)
            faiss.write_index(self.index, index_file)
        else:
            self.index = faiss.read_index(self.index_file)
            self.dimension = self.index.d
        if not os.path.exists(metadata_file):
            self.metadata = []
            json.dump(self.metadata, open(metadata_file, "w+"))
        else:
            self.metadata = json.load(open(self.metadata_file))


    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        Add embeddings and their corresponding metadata to the index.
        """
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top_k closest embeddings and return their metadata and distances.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        results: List[Tuple[Dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # 1 - cosine distance is cosine similarity
            results.append((self.metadata[idx], 1 - float(dist)))
        return results
    
    def save(self):
        faiss.write_index(self.index, self.index_file)
        json.dump(self.metadata, open(self.metadata_file, "w+"))

# ----------------------
# Build Index Function
# ----------------------
def build_faiss_index(
    bullets: List[Dict[str, Any]],  # Each dict should have 'text' and 'session_date'
    user_prefix: str
):
    """
    Given a list of bullet-point records with text and metadata,
    build and return a FAISS index with stored metadata.
    """
    texts = [b['text'] for b in bullets]
    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]
    
    faiss_idx = FaissIndexWithMetadata(user_prefix=user_prefix,
                                       dimension=dim)
    faiss_idx.add(embeddings, bullets)
    faiss_idx.save()

# ----------------------
# Retrieval Function
# ----------------------
def retrieve_bullets(
    user_prefix: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.4
):
    """
    Retrieve the top_k most relevant bullets for a given query.
    Returns a list of metadata dicts with 'text', 'session_date', and 'distance'.
    """
    faiss_idx = FaissIndexWithMetadata(user_prefix=user_prefix)
    q_emb = get_embeddings([query])
    hits = faiss_idx.search(q_emb, top_k)
    results = []
    print(hits)
    for meta, dist in hits:
        entry = meta.copy()
        if dist > threshold:
            results.append(f"Date: {entry["session_date"]}, Content: {entry["text"]}")
    return "\n".join(results)
