# src/ner.py

import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> list:
    """
    Extract named entities (e.g., persons, organizations, dates, etc.) from the text.
    
    Parameters:
        text (str): Input text.
    
    Returns:
        list: A list of tuples (entity_text, entity_label).
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

if __name__ == '__main__':
    test_text = (
        "Joe Biden announced new vaccine initiatives at the White House on March 15, 2023, "
        "in collaboration with Pfizer and Moderna."
    )
    entities = extract_entities(test_text)
    print("Extracted Entities:")
    for ent in entities:
        print(ent)
