# src/linkedin_post.py

from transformers import pipeline

# Initialize the text-generation pipeline with GPT-2.
# (If you must stick with GPT-2 as per your constraints, this remains the model of choice.)
post_generator = pipeline(
    "text-generation",
    model="gpt2"
)

def is_complete(text: str) -> bool:
    """
    Check if the text appears to end with proper sentence-ending punctuation.
    """
    return text.strip().endswith(('.', '!', '?'))

def generate_linkedin_post(text: str, initial_max_length: int = 200, max_iterations: int = 3) -> str:
    """
    Generate a professional LinkedIn post based on the provided news summary using GPT-2.
    The function uses a refined prompt, and if the generated text does not end with proper 
    punctuation (i.e., appears incomplete), it will increment the max_length and try again.
    
    Parameters:
        text (str): The news summary.
        initial_max_length (int): Initial maximum token length for the generated output.
        max_iterations (int): Number of additional iterations if the output seems incomplete.
        
    Returns:
        str: The final LinkedIn post.
    """
    max_length = initial_max_length
    post = ""
    
    # Refined prompt:
    # - Position GPT-2 as a corporate communications expert.
    # - Provide an example style as a guide.
    prompt = (
        "You are an expert in corporate communications. Generate a detailed, professional, and engaging "
        "LinkedIn post that highlights the following details: mention the new policies, discuss their impact "
        "on the economy, and note the mixed reactions in the capital. Do not echo the instructions or the news summary. "
        "For example, your post might be: \"Recent policy changes in the capital have sparked varied responses among industry leaders, "
        "with innovative reforms praised by some and cautioned by others.\" \n\n"
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
        
        # Extract the text after the marker "LinkedIn Post:".
        if "LinkedIn Post:" in full_text:
            post = full_text.split("LinkedIn Post:")[-1].strip()
        else:
            post = full_text
        
        # Optional: Post-process by keeping only the first 2 sentences.
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
    sample_summary = (
        "New policies introduced to improve the economy and bolster social welfare have garnered mixed reactions "
        "in the capital. Critics and supporters alike are debating the potential long-term impacts of these changes."
    )
    
    linkedin_post = generate_linkedin_post(sample_summary, initial_max_length=200, max_iterations=3)
    
    print("Original Text:")
    print(sample_summary)
    print("\nGenerated LinkedIn Post:")
    print(linkedin_post)
