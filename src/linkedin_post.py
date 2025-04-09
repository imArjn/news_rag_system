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

def generate_linkedin_post(text: str, mode: str = "default", include_instructions: bool = True,
                           initial_max_length: int = 250, max_iterations: int = 3) -> str:
    """
    Generate a LinkedIn post based on the provided input.

    Parameters:
        text (str): The content to base the LinkedIn post on.
          - In dynamic mode, text is the direct input (user query plus extra data).
          - In default mode, text is expected to be a retrieved summary from the database.
        mode (str): "dynamic" for user-provided content; "default" for a retrieved summary.
        include_instructions (bool): If True, use a detailed instruction prompt;
                                     if False, use a simpler prompt.
        initial_max_length (int): Initial maximum token length for generation.
        max_iterations (int): Number of regeneration attempts if the output seems incomplete.
        
    Returns:
        str: The final generated LinkedIn post.
    """
    max_length = initial_max_length
    post = ""
    
    # Build the prompt based on mode and whether we want to include full instructions.
    if include_instructions:
        if mode == "dynamic":
            prompt = (
                "Generate a professional and engaging LinkedIn post based on the following news summary. "
                "Your post should capture key details and present them in a polished, insightful style.\n\n"
                f"Input: {text}\n\nLinkedIn Post:"
                "LinkedIn Post:"
            )
        else:
            prompt = (
                "Generate a comprehensive and engaging LinkedIn post based on the following news summary. "
                "Your post should articulate the key facts and provide insights into the implications of the news, "
                "encouraging further reflection among professionals. Do not repeat the prompt or the summary; produce only the final post text.\n\n"
                f"News Summary: {text}\n\nLinkedIn Post:"
            )
    else:
        # Simpler prompt: use only the user-provided content without extra corporate instructions.
        prompt = ("Generate a professional and engaging LinkedIn post based on the following news summary. "
                    "Your post should capture key details and present them in a polished, insightful style.\n\n"
                f"Write a professional LinkedIn post about: {text}. At leaset 2 to 3 sentences.\nLinkedIn Post:")
    
    # Generate text and adjust max_length if the output appears incomplete.
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

        # Extract the generated post by splitting on the marker.
        if "LinkedIn Post:" in full_text:
            post = full_text.split("LinkedIn Post:")[-1].strip()
        else:
            post = full_text

        # Optional: trim output to the first two sentences.
        sentences = post.split('.')
        if len(sentences) > 2:
            post = '.'.join(sentences[:2]).strip() + '.'

        if is_complete(post):
            break
        else:
            max_length += 20
            print(f"Output seems incomplete. Increasing max_length to {max_length} and regenerating...")
    
    return post

if __name__ == '__main__':
    # Test dynamic mode: when extra data is provided.
    dynamic_input = "Who is the French Spider-Man? Additional details: What did he climb?"
    linkedin_post_dynamic = generate_linkedin_post(dynamic_input, mode="dynamic", include_instructions=False,
                                                   initial_max_length=250, max_iterations=3)
    
    print("=== Dynamic Mode Output ===")
    print("Input (User Query + Additional Data):")
    print(dynamic_input)
    print("\nGenerated LinkedIn Post:")
    print(linkedin_post_dynamic)
    
    # Test default mode: when using a retrieved summary.
    default_input = (
        "Alain Robert, known as the 'French Spider-Man', has famously climbed high-rise buildings including the Burj Khalifa and the Empire State Building."
    )
    linkedin_post_default = generate_linkedin_post(default_input, mode="default", include_instructions=False,
                                                   initial_max_length=250, max_iterations=3)
    
    print("\n=== Default Mode Output ===")
    print("Input (Retrieved Summary):")
    print(default_input)
    print("\nGenerated LinkedIn Post:")
    print(linkedin_post_default)
