
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model only once for efficiency.
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """
    Converts the input text into an embedding vector.
    
    Parameters:
        text (str): The text to be encoded.
    
    Returns:
        numpy.ndarray: The vector representation (embedding) of the input text.
    """
    # Check if the text is valid
    if text and isinstance(text, str):
        return model.encode(text)
    return None

if __name__ == '__main__':
    
    sample_text = "Breaking news: Major breakthrough in AI technology."
    embedding_vector = get_embedding(sample_text)
    
    print("Embedding vector obtained:")
    print(embedding_vector)
 
