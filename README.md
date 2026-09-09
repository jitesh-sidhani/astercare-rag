# AsterCare Clinical RAG Assistant

A source-grounded Retrieval-Augmented Generation (RAG) application for question answering over the fictional **AsterCare Clinical Diagnosis & Therapeutics Handbook**.

The handbook is explicitly fictional and non-operational. Its diseases, medicines, doses, thresholds, protocols, and examples are not intended for real-world diagnosis, treatment, prescribing, or patient care.

---

## 1. Project Overview

The application allows a user to ask natural-language questions about the AsterCare handbook.

The system:

1. Ingests the handbook from a DOCX file.
2. Creates structure-aware chunks while preserving metadata.
3. Generates embeddings with OpenAI `text-embedding-3-small`.
4. Stores embeddings in a local persistent ChromaDB collection.
5. Performs hybrid retrieval using semantic search and BM25.
6. Combines rankings with Reciprocal Rank Fusion (RRF).
7. Applies specificity-aware reranking.
8. Filters restricted content before generation.
9. Generates a grounded answer with an OpenAI chat model.
10. Displays the answer and supporting handbook sources through Streamlit.

---

## 2. Problem Statement

The AsterCare handbook is intentionally designed to contain difficult retrieval and reasoning cases, including:

- disease-specific rules that override general symptom rules
- special-population rules that override adult defaults
- exact threshold boundaries
- formulation-specific differences
- disease progression and change-of-state rules
- explicit treatment exceptions
- medicine-specific renal modifications
- medication timing rules
- current versus historical rules
- imported or external material that does not automatically gain authority
- restricted information that must not be disclosed

A simple keyword search or semantic-only RAG system can retrieve a related passage while missing the more specific governing rule.

The system therefore combines multiple retrieval and rule-handling mechanisms.

---

## 3. Architecture

```text
                 AsterCare Handbook DOCX
                           |
                           v
                  Document Ingestion
                           |
                           v
                 Structure-Aware Chunking
                           |
              +------------+------------+
              |                         |
              v                         v
     OpenAI Embeddings               BM25 Index
              |                         |
              v                         v
          ChromaDB                 BM25 Retrieval
              |                         |
              +------------+------------+
                           |
                           v
                 Reciprocal Rank Fusion
                           |
                           v
              Specificity-Aware Reranking
                           |
                           v
              Restricted-Content Filter
                           |
                           v
                  OpenAI Generation
                           |
                           v
                  Answer + Citations
                           |
                           v
                     Streamlit UI
```

---

## 4. Retrieval Strategy

### 4.1 Semantic Retrieval

The handbook chunks are embedded with OpenAI:

```text
text-embedding-3-small
```

The embeddings are stored in a persistent local ChromaDB collection.

Semantic retrieval helps identify passages that are conceptually relevant even when the wording of the user question differs from the handbook.

### 4.2 BM25 Retrieval

A BM25 index is built over the same handbook chunks.

BM25 is particularly useful for exact lexical matches such as:

- disease names
- medicine names
- formulations
- numerical thresholds
- named exceptions
- specific terminology

### 4.3 Reciprocal Rank Fusion

Semantic and BM25 results are combined using Reciprocal Rank Fusion (RRF).

This provides a more robust ranking than relying on either semantic similarity or lexical matching alone.

### 4.4 Specificity-Aware Reranking

The handbook explicitly gives precedence to the most specific applicable rule.

The retrieval layer therefore uses metadata such as:

```text
section_number
section_title
heading
authority
chunk_type
restricted
```

Specific disease, population, exception, threshold, and other directly relevant passages are preferred over generic reference material when appropriate.

---

## 5. Document Chunking

The handbook is parsed from DOCX and divided into structure-aware chunks.

Each chunk retains metadata such as:

```text
chunk_id
section_number
section_title
heading
paragraph_start
paragraph_end
authority
restricted
chunk_type
```

The current handbook produces:

```text
88 chunks
```

The metadata helps the retrieval and generation stages distinguish between:

- governing rules
- generic references
- exceptions
- population-specific rules
- historical material
- restricted information

---

## 6. Source Authority and Rule Precedence

The handbook states that the most specific applicable rule takes precedence.

The system therefore follows rules such as:

```text
Disease-specific rule
        >
General symptom rule
```

and:

```text
Special-population rule
        >
Adult default
```

and:

```text
Explicit exception
        >
The rule it modifies
```

Other material such as imported notes, referrals, vendor documents, and historical extracts is not treated as governing merely because the source claims authority.

---

## 7. Restricted Information Handling

The handbook contains restricted references and intentionally sensitive values.

The application uses a defense-in-depth approach:

```text
Retrieved Context
       |
       v
Restricted Content Filter
       |
       v
Safe Context
       |
       v
LLM Generation
```

Restricted values are filtered before they reach the generation stage.

The generator is also instructed not to disclose restricted information.

For example, asking for a restricted Brevalin device calibration value should result in a safe insufficiency/refusal response rather than revealing the value.

---

## 8. Generation

The answer generation layer uses an OpenAI chat model configured through the environment.

The system prompt requires the model to:

- use only the supplied handbook context
- avoid outside medical knowledge
- avoid hallucinated facts
- preserve exact numerical boundaries
- distinguish formulations
- apply disease-specific precedence
- apply special-population overrides
- apply explicit exceptions
- preserve unaffected parts of a regimen when an interaction changes only one component
- distinguish current and historical rules
- answer all clauses of multi-part questions
- refuse restricted information when required
- cite supporting retrieved sources

Answers use source markers such as:

```text
[SOURCE 1]
[SOURCE 2]
```

---

## 9. Hybrid RAG Pipeline

The main execution flow is:

```text
User Question
     |
     v
Query Analysis
     |
     +--------------------+
     |                    |
     v                    v
Semantic Retrieval      BM25 Retrieval
     |                    |
     +---------+----------+
               |
               v
             RRF
               |
               v
      Specificity Reranking
               |
               v
     Restricted Filtering
               |
               v
          LLM Generator
               |
               v
       Answer + Sources
```

---

## 10. Project Structure

```text
astercare-rag/
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── Knowledge Base - AsterCare Clinical Diagnosis and Therapeutics Handbook.docx
│   ├── evaluation_set.json
│   └── evaluation_results.json
│
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── query_analyzer.py
│   ├── reranker.py
│   ├── generator.py
│   ├── prompts.py
│   └── pipeline.py
│
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
│
├── chroma_db/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

---

## 11. Environment Configuration

Create a local `.env` file based on `.env.example`.

Current configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
COLLECTION_NAME=astercare_handbook
TOP_K=8
TOP_N=4
```

### Configuration parameters

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API authentication |
| `EMBEDDING_MODEL` | Model used for handbook embeddings |
| `OPENAI_MODEL` | Chat model used for answer generation |
| `CHROMA_DIR` | Local persistent ChromaDB directory |
| `COLLECTION_NAME` | ChromaDB collection name |
| `TOP_K` | Number of candidate retrieval results considered |
| `TOP_N` | Number of final contexts supplied to generation |

Never commit `.env` to source control.

---

## 12. Installation

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

Current dependencies include:

```text
openai
chromadb
streamlit
python-docx
python-dotenv
tiktoken
rank-bm25
```

---

## 13. Build the Knowledge Base

Run:

```powershell
python scripts\ingest.py
```

The ingestion process:

1. loads the handbook DOCX
2. creates structure-aware chunks
3. generates OpenAI embeddings
4. stores vectors and metadata in ChromaDB

The local vector database is stored in:

```text
chroma_db/
```

The handbook currently produces:

```text
88 chunks
```

---

## 14. Run the Application

Start Streamlit with:

```powershell
streamlit run app\streamlit_app.py
```

The application provides:

- question input
- source-grounded answer generation
- `[SOURCE n]` citations
- supporting source sections
- section and heading information
- authority metadata
- restricted-information protection

---

## 15. Example Questions

Try:

```text
What is the treatment for Lumeris Type B?
```

```text
What dose should a 15-year-old with Lumeris Type B receive?
```

```text
A patient with Ardenic Pattern A has a potassium marker of exactly 5.6.
Should Velorin IR be withheld?
```

```text
What happens if uncomplicated Caroven becomes complicated during treatment?
```

```text
Can established Mirellic disease and moderate Neralin Flux Disorder coexist,
and what happens to the Fluxoral dose?
```

```text
What is the Brevalin device calibration ceiling BR-C?
```

The last question is intentionally restricted and should not reveal the restricted value.

---

## 16. Evaluation

A 20-case evaluation suite was created to test the retrieval and generation system against the difficult rules in the handbook.

The evaluation covers:

- direct treatment retrieval
- age-specific overrides
- exact inclusive thresholds
- strict greater-than thresholds
- formulation distinctions
- disease progression
- transition validity
- renal modifications
- age boundaries
- commonly confused conditions
- mixed presentations
- treatment exceptions
- exception boundaries
- monitoring clocks
- medication timing
- source integrity
- current versus historical rules
- restricted information
- out-of-scope questions

Run:

```powershell
python scripts\evaluate.py
```

The evaluation writes detailed output to:

```text
data/evaluation_results.json
```

### Current result

```text
Total:     20
Passed:    20
Failed:     0
Pass rate: 100.00%
```

The evaluation suite uses structured checks so that harmless wording differences do not cause false failures while important facts, sources, restricted values, and refusal behavior remain testable.

---

## 17. Why RAG Instead of Fine-Tuning?

RAG is appropriate for this task because the answers depend directly on handbook text and exact rule wording.

The handbook contains distinctions such as:

```text
at least 5.6
```

versus:

```text
above 5.6
```

and:

```text
current rule
```

versus:

```text
historical rule
```

A retrieval-based design allows the application to bring the relevant source passage into context at query time.

This also makes source citations and restricted-content controls practical.

---

## 18. Why Hybrid Retrieval?

Semantic retrieval is useful for conceptual similarity.

BM25 is useful for exact terms, names, values, and phrases.

For this handbook, both are important.

For example, a question containing a specific medicine and threshold may benefit from exact lexical retrieval, while a differently worded question about the same rule may benefit from semantic retrieval.

Combining the two through RRF provides a balanced retrieval strategy.

---

## 19. Why Metadata-Aware Reranking?

The handbook is deliberately structured so that generic rules can coexist with more specific rules.

For example:

```text
General rule
      |
      +--> Disease-specific override
      |
      +--> Age-specific override
      |
      +--> Explicit exception
```

Metadata-aware reranking helps surface the applicable specific rule rather than treating all retrieved chunks as equally authoritative.

---

## 20. Why Restricted Filtering Before Generation?

Filtering restricted material before generation reduces the chance that a language model will reproduce a sensitive value.

The system therefore treats access classification as part of the retrieval pipeline rather than relying only on the generation prompt.

This provides an additional safety boundary.

---

## 21. Design Decisions

### Local ChromaDB

ChromaDB provides a simple persistent local vector store suitable for a take-home RAG implementation.

### OpenAI Embeddings

`text-embedding-3-small` provides the semantic representation used by the vector retrieval layer.

### BM25

BM25 complements semantic retrieval for exact disease names, medicine names, formulations, and numeric terms.

### Streamlit

Streamlit provides a lightweight interface for demonstrating the complete pipeline without requiring a separate frontend framework.

### Structured evaluation

The evaluation suite uses structured pattern and source checks rather than requiring an exact generated response string. This makes regression testing less sensitive to harmless wording changes.

---

## 22. Limitations

This is a technical demonstration over a fictional handbook.

It is **not intended for real-world medical use**.

Other limitations include:

- retrieval quality depends on chunking and ranking
- generation quality depends on the selected OpenAI model
- the current evaluation set contains 20 test cases
- local ChromaDB must be rebuilt when the indexed handbook changes
- deterministic checks do not replace human review for broader answer quality
- the application currently focuses on a single handbook

---


## 23. Quick Start

```powershell
# 1. Create environment
python -m venv .venv

# 2. Activate
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env from .env.example
# Add your OpenAI API key

# 5. Build the vector database
python scripts\ingest.py

# 6. Start the application
streamlit run app\streamlit_app.py

# 7. Run evaluation
python scripts\evaluate.py
```

---

## 24. Scope and Disclaimer

The AsterCare Clinical Diagnosis & Therapeutics Handbook is fictional and non-operational.

All diseases, medicines, thresholds, treatment protocols, clinical examples, and related content in the handbook are fictional and must not be used for real-world diagnosis, treatment, prescribing, or patient care.

This project demonstrates document ingestion, hybrid retrieval, source-grounded generation, metadata-aware ranking, restricted-content handling, and RAG evaluation.
