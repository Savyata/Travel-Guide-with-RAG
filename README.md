# Travel-Guide-with-RAG
## Nepal Travel Assistant - RAG System

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer tourism-related questions about Nepal using local context and custom datasets. It compares the performance of three open-source Large Language Models (LLMs) under resource-constrained environments.



## Objectives

- Build a RAG pipeline to answer domain-specific travel questions.
- Evaluate and compare the performance of three LLMs:
  - FLAN-T5 Large
  - FLAN-T5 Base
  - TinyLLaMA 1.1B
- Support real-time question answering using a custom dataset about Nepal.

---

## Dataset Overview

The dataset was compiled from trusted sources including travel blogs, government websites, hotel reviews, and forums. It includes:

- Traditional foods
- Accommodation and hotel costs
- Visa process for U.S. citizens
- Flight prices from the U.S.
- Major festivals
- Local transportation options
- Everest Base Camp trek guide
- Top destinations and trekking routes
- Packing checklist
- Budget accommodations

All text files were chunked using regex or fixed-size slicing to ensure semantic coherence and optimize retrieval accuracy.


## Architecture: RAG System

1. **Chunking & Preprocessing**  
   Raw `.txt` files were cleaned and chunked using regex or token size constraints.

2. **Embedding**  
   SentenceTransformer `all-MiniLM-L6-v2` was used to vectorize text chunks (shape: `(104, 384)`).

3. **Indexing**  
   FAISS was used to store and retrieve top-`k` relevant chunks for any user query (k=4).

4. **LLM-Based Generation**  
   Three different LLMs generated answers conditioned on retrieved chunks:
   - Google FLAN-T5 (Large & Base)
   - TinyLLaMA-1.1B

5. **Evaluation**  
   - 13 domain-specific questions tested.
   - Each model was scored (0–5) based on relevance.
   - Analysis included hallucination, reasoning, and fluency observations.


## Models Compared

| Model         | Type          | Parameters | Strengths                        | Weaknesses                          |
|---------------|---------------|------------|----------------------------------|--------------------------------------|
| FLAN-T5 Large | Encoder-Decoder | ~780M     | Concise and factually correct    | Short/incomplete outputs             |
| FLAN-T5 Base  | Encoder-Decoder | ~250M     | Low hallucination, fast runtime | Lacks detail, generic answers        |
| TinyLLaMA     | Decoder-Only    | ~1.1B     | Highly detailed, fluent answers | Prone to hallucination and verbosity |


## Sample Domain Questions

1. What is the must-try traditional food in Nepal?
2. How much do hotels cost in Kathmandu?
3. What is the visa process for U.S. citizens?
4. What is the average flight cost from the U.S.?
5. Which major festivals should tourists plan around?
6. What are the local transport options?
7. What is a 10-day Everest Base Camp itinerary?
8. Top 3 destinations in Nepal?
9. Best trekking routes?
10. Beginner-friendly treks?
11. Hardest treks for experts?
12. What to pack for Himalayan treks?
13. Cheapest places to stay in Nepal?

---

## Results

- **Average Scores (out of 5):**
  - TinyLLaMA: **3.77**
  - FLAN-T5 Large: **2.23**
  - FLAN-T5 Base: **1.85**

- TinyLLaMA gave the most detailed answers, while FLAN-T5 Base gave safer but less informative responses. FLAN-T5 Large gave brief but relevant responses.


## Evaluation Criteria

- **Relevance** to query
- **Factual Accuracy**
- **Reasoning & Inference**
- **Conciseness**
- **Fluency & Structure**

***Manual and auto scoring were applied.***



## Setup

# Install required packages
pip install faiss-cpu transformers sentence-transformers pandas

# Download models inside notebook
from transformers import pipeline
pipeline("text2text-generation", model="google/flan-t5-base")

Note: Two notebooks were used:

NLP_Final_Project_part1_SR.ipynb for chunking, embedding, FAISS.

NLP_Final_Project_Part2_SR.ipynb for inference and evaluation.

## Future Work
- Add hallucination filters
- Try new models (Phi-3, Mistral, LLaMA-3)
- Use hybrid model routing based on query type
- Add visual map-based travel plans

## Author
Savyata Regmi

MS in Data Science

University of New Haven

Spring 2025 | DSCI 6004 NLP Final Project
