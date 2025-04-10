# src/data_preprocessing.py

import pandas as pd
import numpy as np
from src.embedding import get_embedding  # Import the function from embedding.py
from src.search import build_index, search # Importing the function from the search.py


def load_data(file_path):
    """
    Load data from the JSON file.
    Assumes that the file is line-delimited.
    """
    try:
        data = pd.read_json(file_path, lines=True)
        print("Data loaded successfully! Shape:", data.shape)
    except Exception as e:
        print("Error loading data:", e)
        data = None
    return data

def clean_data(data):
    """
    Clean the data. This includes removing duplicates.
    """
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    final_rows = data.shape[0]
    print(f"Removed {initial_rows - final_rows} duplicate rows.")
    return data

def add_embeddings(data, text_column='short_description'):
    """
    Add a new column 'embedding' to the DataFrame by applying get_embedding()
    on the specified text column.
    """
    if text_column not in data.columns:
        print(f"Column '{text_column}' not found in data.")
        return data

    # Create the 'embedding' column by applying get_embedding() on each row's text
    data['embedding'] = data[text_column].apply(lambda text: get_embedding(text))
    return data

if __name__ == '__main__':
    # Path to the full dataset file
    file_path = "data/sample.json"
    data = load_data(file_path)
    if data is not None:
        data = clean_data(data)
        data = add_embeddings(data, text_column='short_description')
        
        # Convert the list of embeddings to a NumPy array
        embeddings_list = data['embedding'].tolist()
        embeddings_np = np.vstack(embeddings_list)
        print("Embeddings array shape:", embeddings_np.shape)
        
        # Build the FAISS index with the embeddings
        index = build_index(embeddings_np)
        
        # Create a query embedding
        query_text = "Latest updates on political news"  # Example query for testing
        query_embedding = get_embedding(query_text)
        
        # Search for top 5 nearest neighbors
        indices, distances = search(index, query_embedding, top_k=5)
        print("Search results:")
        print("Indices:", indices)
        print("Distances:", distances)

