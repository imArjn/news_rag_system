from transformers import pipeline

# Initialize the summarization pipeline using facebook/bart-large-cnn
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary_local(text, default_max=40, default_min=20):
    """
    Generate a summary for the given text using BART.
    For very short inputs, if the text is already concise, return it or do a slight rephrase.
    """
    words = text.split()
    input_length = len(words)
    
    # If the text is very short, just rephrase slightly or return a modified version
    if input_length < 50:
        # Option 1: Just return a rephrased version. You could also try summarizer with more constrained parameters.
        summary_list = summarizer(
            text,
            max_length=default_max,
            min_length=default_min,
            num_beams=3,
            do_sample=True,
            temperature=0.8
        )
        return summary_list[0]['summary_text']
    else:
        # For longer texts, you can adjust dynamically.
        max_length = min(100, input_length + 20)
        min_length = min(40, input_length)
        summary_list = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=3,
            do_sample=True,
            temperature=0.8
        )
        return summary_list[0]['summary_text']

if __name__ == '__main__':
    sample_text = (
        "Breaking news: A major political development occurred today in the capital as government officials announced an ambitious set of reforms aimed at tackling the economic slowdown and improving public welfare. The new measures include a comprehensive tax reform package intended to boost small and medium enterprises, increased funding for social programs such as healthcare, education, and affordable housing, and enhanced support for unemployed citizens. Economic experts have largely welcomed the proposals, suggesting that these initiatives could stimulate growth and generate long-term benefits, though some remain skeptical about the potential rise in national debt."
        "Critics argue that while the reforms are well-intentioned, they may lead to unintended consequences such as inflation or reduced investor confidence if not implemented carefully. Public opinion is also divided: while many citizens are hopeful that the changes will lead to a more inclusive economy, there is also concern over the potential disruption to existing social programs and the impact on everyday life. In response, government spokespersons have promised a transparent rollout and continuous monitoring of the reforms’ progress, assuring the public that any adverse effects will be swiftly addressed. This decisive move marks a turning point in the nation’s economic policy, and all eyes are now on the capital to see whether these bold reforms will bring about the change that so many hope for."

    )
    summary = generate_summary_local(sample_text)
    print("Original Text:")
    print(sample_text)
    print("\nGenerated Summary:")
    print(summary)
