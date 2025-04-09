# src/linkedin_post.py

from transformers import pipeline

# Initialize the text-generation pipeline with GPT-2.
post_generator = pipeline(
    "text-generation",
    model="gpt2"
)

def is_complete(text: str) -> bool:
    """
    Check if the text appears to end with proper sentence-ending punctuation.
    """
    return text.strip().endswith(('.', '!', '?'))

def generate_linkedin_post(text: str, mode: str = "default", initial_max_length: int = 250, max_iterations: int = 3) -> str:
    """
    Generate a LinkedIn post based on the provided input.
    
    Parameters:
        text (str): The content to base the LinkedIn post on.
            - In dynamic mode, 'text' should be a combination of the user query and extra details.
            - In default mode, 'text' is a news summary retrieved from the database.
        mode (str): "dynamic" to generate from user-provided combined input; 
                    "default" to generate from a retrieved summary.
        initial_max_length (int): Initial maximum token length for generation.
        max_iterations (int): Maximum attempts if the output seems incomplete.
    
    Returns:
        str: The generated LinkedIn post.
    """
    max_length = initial_max_length
    post = ""
    
    if mode == "dynamic":
        # In dynamic mode, use a prompt that is tailored to the topics provided.
        prompt = (
            f"Write a detailed, professional, and engaging LinkedIn post about the following topics: {text}. "
            "The post should be at least three sentences long, offering insights and a clear call-to-action."
        )
    else:
        # In default mode, assume text is a retrieved news summary.
        prompt = (
            f"Based on the following news summary, generate a comprehensive and engaging LinkedIn post that clearly reflects its key points and implications. "
            f"News Summary: {text}\n\nLinkedIn Post:"
        )
    
    for i in range(max_iterations):
        output = post_generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            truncation=True
        )
        full_text = output[0]['generated_text'].strip()
        
        # In default mode, try to extract the text after our marker (if present)
        if mode == "default" and "LinkedIn Post:" in full_text:
            post = full_text.split("LinkedIn Post:")[-1].strip()
        else:
            post = full_text
        
        # Optional: Ensure a minimum number of sentences; here, we require at least 3.
        sentences = [s.strip() for s in post.split('.') if s.strip()]
        if len(sentences) >= 3 and is_complete(post):
            break
        else:
            max_length += 20
            print(f"Output seems incomplete. Increasing max_length to {max_length} and regenerating...")
    
    return post

if __name__ == '__main__':
    # Test dynamic mode: user provides extra data (e.g., "I want a LinkedIn post on Joe Biden. Additional details: Joe Biden and lady Biden")
    dynamic_input = "I want a LinkedIn post on Joe Biden. Additional details: Joe Biden and lady Biden."
    linkedin_post_dynamic = generate_linkedin_post(dynamic_input, mode="dynamic", initial_max_length=250, max_iterations=3)
    
    print("=== Dynamic Mode Output ===")
    print("Input (User Query + Additional Data):")
    print(dynamic_input)
    print("\nGenerated LinkedIn Post:")
    print(linkedin_post_dynamic)
    
    # Test default mode: using a retrieved summary (simulate a summary from the database)
    default_input = (
        "Joe Biden recently announced new vaccine initiatives and measures to bolster public health, leading to mixed reactions among experts and the public."
    )
    linkedin_post_default = generate_linkedin_post(default_input, mode="default", initial_max_length=250, max_iterations=3)
    
    print("\n=== Default Mode Output ===")
    print("Input (Retrieved Summary):")
    print(default_input)
    print("\nGenerated LinkedIn Post:")
    print(linkedin_post_default)
