# src/search.py

import faiss
import numpy as np

def build_index(embeddings):
    """
    Build a FAISS index from the provided embeddings.

    Parameters:
        embeddings (numpy.ndarray): A 2D array of shape (n_samples, embedding_dimension)

    Returns:
        index: A FAISS index object with the embeddings added.
    """
    # Determine embedding dimension
    dimension = embeddings.shape[1]
    # Create a FAISS index using L2 (Euclidean) distance metric
    index = faiss.IndexFlatL2(dimension)
    # Add embeddings to the index
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

def search(index, query_embedding, top_k=5):
    """
    Search the FAISS index for the top_k nearest neighbors to the query embedding.
    
    Parameters:
        index: A FAISS index object.
        query_embedding (numpy.ndarray): The embedding of the query text, as a 1D array.
        top_k (int): Number of nearest neighbors to retrieve.
    
    Returns:
        indices: Indices of the retrieved nearest neighbors.
        distances: Distance values corresponding to the retrieved neighbors.
    """
    # Ensure query_embedding is a 2D array as FAISS expects shape (1, dimension)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances

if __name__ == '__main__':
    # test purposes:
    
    # Create dummy embeddings: For example, 1000 samples of 384-dimensional vectors
    dummy_embeddings = np.random.rand(1000, 384).astype('float32')
    index = build_index(dummy_embeddings)
    
    # Create a dummy query embedding (384-dimensional)
    query_embedding = np.random.rand(384).astype('float32')
    top_k = 5
    indices, distances = search(index, query_embedding, top_k)
    
    print("Retrieved indices:", indices)
    print("Distances:", distances)


