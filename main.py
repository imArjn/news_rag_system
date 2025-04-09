# main.py

from src.data_preprocessing import load_data, clean_data, add_embeddings
from src.search import build_index, search
from src.embedding import get_embedding
from src.summarization import generate_summary_local
from src.router import route_request
from src.linkedin_post import generate_linkedin_post
import numpy as np

def interactive_query():
    """
    Prompt the user to enter a query and, if additional data is provided,
    collect that extra data.
    
    Returns:
        user_query (str): The main query text.
        data_provided (bool): True if extra data is provided.
        extra_data (str): The additional content, if provided, else an empty string.
    """
    user_query = input("Enter your query (what would you like to do?): ").strip()
    
    extra_response = input("Is additional data provided for LinkedIn post generation? (Y/N): ").strip().lower()
    
    if extra_response == "y":
        data_provided = True
        extra_data = input("Please enter the additional data: ").strip()
    else:
        data_provided = False
        extra_data = ""
    
    return user_query, data_provided, extra_data

def main():
    # Data loading and preprocessing (as before)
    file_path = "data/sample.json"  # or "data/sample.json" for development
    data = load_data(file_path)
    if data is None:
        print("Failed to load data!")
        return

    data = clean_data(data)
    data = add_embeddings(data, text_column='short_description')
    
    embeddings_list = data['embedding'].tolist()
    embeddings_np = np.vstack(embeddings_list)
    index = build_index(embeddings_np)
    
    # Interactive query input with additional data collection.
    user_query, data_provided, extra_data = interactive_query()
    
    # Use the router to determine which agent to use.
    agent = route_request(user_query, data_provided)
    print("Router directs the query to:", agent)
    
    if agent == "Agent2":
        # Process as a news retrieval & summarization request.
        query_embedding = get_embedding(user_query)
        top_k = 3
        indices, distances = search(index, query_embedding, top_k)
        print("Retrieved Articles (by indices):", indices)
        for idx in indices[0]:
            article = data.iloc[idx]
            headline = article.get("headline", "No Headline")
            short_description = article.get("short_description", "")
            summary = generate_summary_local(short_description)
            print("\n--- Retrieved Article ---")
            print("Headline:", headline)
            print("Original Short Description:", short_description)
            print("Summary:", summary)
    elif agent == "Agent3":
        # Process as a LinkedIn post generation request.
        # If extra data was provided, use it; otherwise, use the query.
        if extra_data:
            input_text = extra_data
        else:
            input_text = user_query
        linkedin_post = generate_linkedin_post(input_text)
        print("\n--- Generated LinkedIn Post ---")
        print(linkedin_post)

if __name__ == '__main__':
    main()
