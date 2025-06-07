Read the report [markdown](https://github.com/Jennifer-J-Li/relevance_gap_rag_llm/tree/main/RAG_LLM_for_Ads_Relevance_Evaluation_markdown) file for script and results.

## Objective
"Difficult queries" are tail (queries least frequently searched by shoppers) search queries in specific product lines. Ads discoverbility in such queries is challenging. 
Scientists tackle the retrieval by enriching ads engaged from similar queries, or matching similar non-ads products.

## Overview

The core workflow involves:
1.  **Data Loading & Preparation:** Ingests product data and query-product interaction examples (e.g., use the [ESCI](https://github.com/amazon-science/esci-data) dataset as simulation).
2.  **Vector Store Population:** Creates embeddings for product descriptions (chunked for better retrieval) and stores them in a ChromaDB vector database.
3.  **Initial Scoring & Difficult Query Identification:**
    *   Scores products against original user queries using cosine similarity of their embeddings.
    *    Simulated "difficult queries" based on criteria like a mix of relevant (Exact/Substitute) and irrelevant (Complement/Irrelevant) products.
4.  **RAG-based Relevance Refinement for Difficult Queries:**
    *   **Engagement Simulation:** Simulates user engagement with top-ranked "Exact" (E) items.
    *   **Content Condensation:** Uses an LLM (Mistral via Ollama) to condense the descriptions of these engaged items, focusing on query-relevant details.
    *   **Query Rewriting:** Employs the LLM to rewrite the original user query, incorporating implicit intent derived from the condensed engaged items.
    *   **Re-evaluation:**
        *   Calculates new cosine similarity scores for filtered (Complement/Substitute - C/S) ads against the rewritten query.
        *   Uses the LLM to perform a direct relevance judgment (Yes/No with explanation) for each chunk of the filtered ads against the rewritten query.
        *   Updates relevance scores for engaged items against the rewritten query.
5.  **Output & Logging:** Generates scored results, raw LLM outputs for debugging, and logs parsing errors.
