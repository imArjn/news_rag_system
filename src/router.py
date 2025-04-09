# src/router.py
import re

def route_request(user_query: str, data_provided: bool = False) -> str:
    """
    Routes the user query to the appropriate agent.
    
    Parameters:
      user_query (str): The text input or query provided by the user.
      data_provided (bool): If True, indicates that the user has already provided
                            specific data for generating a LinkedIn post.
                            
    Returns:
      str: Returns "Agent3" if data is provided (LinkedIn post generation);
           Otherwise returns "Agent2" for news retrieval & summarization.
    """
    # Option 1: If data is provided (e.g., user has given news content), route to Agent3.
    if data_provided:
        return "Agent3"
    
    # Option 2: If the user_query mentions "LinkedIn" explicitly, you can decide to use Agent3
    # (for example, if the user says "generate a LinkedIn post about ...")
    pattern = r'\blink(?:ed)?\s*in\b'
    if re.search(pattern, user_query, re.IGNORECASE):
        return "Agent3"
    
    # Option 3: Otherwise, assume it's a request for news, so route to Agent2.
    return "Agent2"


if __name__ == '__main__':
    # Testing the router with various examples
    
    test_queries = [
        ("Show me latest political news", False),
        ("Generate a LinkedIn post about the recent economic downturn", False),
        ("Here's my news data: [sample text]", True),
        ("I need a LinkedIn post for the new product launch", False),
    ]
    
    for query, data_flag in test_queries:
        routed_agent = route_request(query, data_flag)
        print(f"Query: {query}\nData provided: {data_flag}\n--> Routed to: {routed_agent}\n")
