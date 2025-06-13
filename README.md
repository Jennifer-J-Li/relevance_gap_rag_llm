Read the report [markdown](https://github.com/Jennifer-J-Li/relevance_gap_rag_llm/tree/main/RAG_LLM_for_Ads_Relevance_Evaluation_markdown) file for script and results.

## Objective

**Business Problem:** Identified the limitations of ads relevance model in tail queries, which are usually "difficult queries" least frequently searched by shoppers. In these queries there were no ads coverage or ads engaments. 

**Root cause:** Although Ads retrieval algorithms return an adeqaunt number of relevant candidates in such serach traffic, relevance model has limitation in inferring the implicit shopping intents from these queries, filtering out 50% - 80% candidates by flagging them irrelevant. 

**Solution:** In some of these queries, shopper engaged with non-ads products. LLMs (Mistral) was adopted to 1) summarize the product descriptions, and 2) expand the original queries with implicit shopping intents, and lastly 3) evaluate ads relevance against the "rewritten" queries.

**Deliverable:**
* Sized the revenue gain in such queries assuming a more intelligent relevance model.
* Before relevance model can be improved, identified a workaround by loosing relevance score threshold, which was optimized using ROC-AUC so that we can achieve maximum recall while retaining reasonable false positive rate.

**NOTE:**
This project use simulated data to showcase the problem solving and techniques as descripbed above, so final result numbers are also simulated.


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
