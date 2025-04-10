# src/hybrid_search.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.search import search  # FAISS-based semantic search import

class HybridSearch:
    def __init__(self, data, tfidf_field="short_description"):
        """
        Initialize the hybrid search object.
        
        Parameters:
            data (pandas.DataFrame): The dataset containing the articles.
            tfidf_field (str): The field to use for keyword-based retrieval.
        """
        self.data = data
        self.tfidf_field = tfidf_field
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Fit TF-IDF on the document texts.
        self.doc_texts = data[tfidf_field].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
    
    def search(self, query: str, index, top_k: int = 5, alpha: float = 0.5) -> list:
        """
        Perform hybrid search combining semantic and keyword-based retrieval.
        
        Parameters:
            query (str): The user query.
            index: The FAISS index built from semantic embeddings.
            top_k (int): The number of top articles to return.
            alpha (float): Weight parameter between keyword (TF-IDF) and semantic search.
                           0.0 means only keyword search, 1.0 means only semantic search.
        
        Returns:
            List of document indices (e.g., sorted by combined relevance score).
        """
        # 1. Semantic Search Score (using your existing search function)
        #    Assume search(index, query_embedding, top_k) returns (indices, distances)
        from src.embedding import get_embedding
        query_embedding = get_embedding(query)
        semantic_indices, semantic_distances = search(index, query_embedding, top_k=top_k*3)
        
        # Convert distances to similarities (note: lower distance = more similar)
        # Here, we assume a simple inversion: semantic_similarity = 1 / (1 + distance)
        semantic_similarities = 1 / (1 + semantic_distances.astype(np.float32))
        semantic_similarities = semantic_similarities[0]  # 1D array
        
        # 2. Keyword-based Search Score via TF-IDF
        query_tfidf = self.vectorizer.transform([query])
        keyword_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # 3. Combine the scores.
        # For each candidate index, get the semantic and keyword score, and do a weighted sum.
        combined_scores = {}
        # Let's consider candidates from the semantic search result
        for idx, sem_sim in zip(semantic_indices[0], semantic_similarities):
            key_sim = keyword_similarities[idx]
            # Combine with weight alpha for semantic, (1-alpha) for keyword.
            combined_score = alpha * sem_sim + (1 - alpha) * key_sim
            combined_scores[idx] = combined_score
        
        # Sort candidate indices by combined score in descending order.
        sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
        # Return top_k indices.
        return sorted_indices[:top_k]
