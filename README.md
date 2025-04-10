# News Article RAG System with LinkedIn Post Generator

This repository implements a Retrieval-Augmented Generation (RAG) system for a news application. The system can:
- Retrieve relevant news articles from the [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- Generate concise, contextual summaries from the retrieved articles using a pre-trained summarization model.
- Produce professional LinkedIn posts based on either a retrieved summary or user-provided extra data.

Additionally, optional features include hybrid search (combining semantic and keyword-based retrieval) and Named Entity Recognition (NER) using spaCy to improve relevance and extract key entities.

---

## Overview

This project is organized into three major parts:

### Part 1: RAG System Development
1. **Knowledge Base Creation:**  
   - Extract and preprocess news articles from the Kaggle dataset.
   - *Files:*  
     - `scripts/create_sample.py` — creates a sample dataset (e.g., 1000 rows) from the full dataset.
     - `src/data_preprocessing.py` — loads, cleans, and attaches embeddings to the dataset.
2. **Vector Database & Retrieval:**  
   - Compute embeddings for each article’s short description using Sentence Transformers.
   - Build a FAISS index for semantic search.
   - *Files:*  
     - `src/embedding.py`  
     - `src/search.py`
3. **LLM-Based Summarization:**  
   - Summarize the retrieved articles using a pre-trained model (facebook/bart-large-cnn).
   - *File:*  
     - `src/summarization.py`

### Part 2: Additional Features (Optional)
1. **Hybrid Search:**  
   - Combine semantic search (using FAISS) with keyword-based search (using TF-IDF) for better retrieval accuracy.
   - *File:*  
     - `src/hybrid_search.py`
2. **Named Entity Recognition (NER):**  
   - Extract key entities (names, dates, locations) from text using spaCy.
   - *File:*  
     - `src/ner.py`

### Part 3: Agent System
1. **Agent 1 (Router):**  
   - Routes user queries based on content and extra data.
   - *File:*  
     - `src/router.py`
2. **Agent 2 (Retrieval & Summarization):**  
   - Retrieves and summarizes the top _k_ relevant articles.
   - *File:*  
     - `src/agent2.py`
3. **Agent 3 (LinkedIn Post Generator):**  
   - Generates a professional LinkedIn post from either a retrieved summary or user-provided content.
   - *File:*  
     - `src/linkedin_post.py`
4. **Main Integration:**  
   - Ties all agents together and provides an interactive command-line interface.
   - *File:*  
     - `main.py`

**Workflows:**
- **General News Request:**  
  User’s query (without extra data) → Agent 1 routes to Agent 2 → Relevant articles are retrieved & summarized.
- **LinkedIn Post with Extra Data:**  
  User provides extra data → Agent 1 routes directly to Agent 3 → Post is generated directly from the combined input.
- **LinkedIn Post on Specific Event (No Extra Data):**  
  User’s query → Agent 1 routes to Agent 3 → Agent 2 retrieves & summarizes articles → Summary is passed to Agent 3 for post generation.

---

## Install Dependencies
pip install -r requirements.txt

## Usage Instructions
#### Creating a Sample Dataset
#### Use the provided script to create a smaller sample from the full dataset:
-`python scripts/create_sample.py`

### Make sure your sample dataset (data/sample.json) is in the correct folder. For a full run, you can switch to data/Dataset.json in main.py.

-`python main.py`

## Four Methods to try to test the system after running `python main.py`
## Testing Scenarios

You can test the functionality of the system using the following scenarios:

1. **Test LinkedIn Post Generation with Extra Data (Dynamic Mode):**
   - **Input Example:**  
     Query: `"I want a LinkedIn post on Joe Biden"`  
     When prompted for additional data, answer **Y** and then provide:  
     `"Also highlight his vaccination efforts and mention Dr. Jill Biden."`
   - **Expected Behavior:**  
     The system combines the main query and extra data and directly generates a detailed LinkedIn post using dynamic mode, bypassing database retrieval.

2. **Test LinkedIn Post Generation without Extra Data (Default Mode):**
   - **Input Example:**  
     Query: `"I want a LinkedIn post on Joe Biden and vaccines"`  
     When prompted for additional data, answer **N**.
   - **Expected Behavior:**  
     The system retrieves the top relevant articles from the dataset, summarizes them to form a combined summary, and then generates a LinkedIn post based on that summary.  
     The console will also display details (headlines, short descriptions, and individual summaries) of the retrieved articles.

3. **Verify Retrieval and Summarization (Agent 2 Debugging):**
   - **Input Example:**  
     Query: `"Tell me about French Spider-Man"`  
     Answer **N** so that the retrieval mechanism is triggered.
   - **Expected Behavior:**  
     The system displays debug information showing the retrieved article indices, headlines, short descriptions, and generated summaries. This confirms that the retrieval & summarization (Agent 2) component is functioning properly.

4. **Test the Router Logic Independently:**
   - **Input Examples:**
     - **Case A:**  
       Query: `"Generate a LinkedIn post on tech news"`  
       Expected to be routed to Agent 3.
     - **Case B:**  
       Query: `"Show me the latest political news"`  
       Expected to be routed to Agent 2.
   - **Expected Behavior:**  
     The console output should include debug statements from the router indicating whether the query is directed to Agent 2 or Agent 3. This confirms that the routing logic correctly distinguishes between requests that require direct LinkedIn post generation and those that require news retrieval and summarization.

## Note on the Generation Model

This project uses GPT‑2 for text generation. While GPT‑2 is a robust, freely available model, its outputs may sometimes be generic or less detailed than desired—especially for highly specific or context-rich generation tasks. For improved quality, consider exploring larger models (e.g., GPT‑J 6B, GPT‑Neo) or fine-tuning a model on domain-specific data in the future.

## Thank You 
