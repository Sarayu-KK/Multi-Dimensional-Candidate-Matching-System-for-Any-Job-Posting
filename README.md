# Multi-Dimensional-Candidate-Matching-System-for-Any-Job-Posting
A Multi-Dimensional Candidate Matching System that can analyze, retrieve, evaluate, and rank candidates for any job posting, using both structured and unstructured data.

SETUP AND INSTALLATION
1. Clone the Repository:
    git clone [(https://github.com/Sarayu-KK/Multi-Dimensional-Candidate-Matching-System-for-Any-Job-Posting/edit/main/README.md)]
    cd [Multi-Dimensional Candidate Matching System for Any Job Posting]

2. Create and Activate Environment:
    conda create -n candmatch python=3.10
    conda activate candmatch

3. Install Dependencies:
   
    pip install -r requirements.txt
  

4.  Set API Key: Obtain an API key from OpenRouter. Set it as an environment variable or enter it in the Streamlit sidebar:
   
    set OPENROUTER_API_KEY=sk-or-v1-...

5. Run the project

Execute the Streamlit application from your terminal:
streamlit run app.py

SYSTEM ARCHUTECTURE AND WORKFLOW

1. Data Flow Overview
This system works in 4 stages , which are:
a.Input
b.Extraction
c.Matching
d.Ranking

2. Extraction stage

Model Used:openai/gpt-3.5-turbo (via OpenRouter)
Purpose:To categorize unstructured job description text into three structured clusters: must_have, important, and nice_to_have.
Prompt used: The prompt explicitly forces the LLM to return a clean JSON object for reliable programmatic parsing.

3. Semantic Matching and Scoring

Embedding Model:openai/text-embedding-3-large 
Chunking:The resume is tokenized into sentences, which are grouped into blocks of 3 sentences (chunks) to capture context.
Cosine Similarity: The embedding of each skill phrase is compared against the embeddings of all resume chunks to find the maximum similarity score
Weighted Scoring Formula: The final candidate score is the sum of weighted semantic matches. The weights are:
    * **Must-Have:** $\text{Score}_{\text{final}} = \sum (\text{Sim}_{\text{max}} \times 4.0)$
    * **Important:** $\text{Score}_{\text{final}} = \sum (\text{Sim}_{\text{max}} \times 1.5)$
    * **Nice-to-Have:** $\text{Score}_{\text{final}} = \sum (\text{Sim}_{\text{max}} \times 1.0)$


4. Ranking and Explainability

Candidates are ranked purely by their overall_score. And  finally a pdf report can be downloaded which explains the comarisons between various resume.
