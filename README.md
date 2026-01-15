# BirdSQL Hybrid NLQ System

A hybrid machine learning system that combines vector search with LLM-powered SQL generation for natural language to SQL query translation. This approach handles both semantic queries ("young clients from Prague") and exact filters (amount > 100,000) using the Bird-Bench dataset.

## Overview

Traditional text-to-SQL systems struggle with:
- **Semantic/fuzzy terms**: "young", "expensive", "popular" 
- **Text variations**: "Prague" vs "Praha" vs "Prage"
- **Exact comparisons**: numeric filters, date ranges

Our **hybrid approach** solves this by:
1. Using **vector search** for semantic/fuzzy matching on descriptive columns
2. Using **SQL filters** for exact numeric/categorical comparisons
3. Combining both with intelligent JOIN generation

## Prerequisites

- Python 3.8+
- Conda/Miniconda
- Streamlit
- GROQ API Key
- OpenAI API Key

## Installation

### 1. Download Dataset

Download the Bird-Bench dataset from [bird-bench.github.io](https://bird-bench.github.io) and extract it to your project directory.

### 2. Setup Data Directory

Organize your data folder with the following structure:

```
data/
├── dev_20240627/
└── train/
```

### 3. Configure Environment

**Install Dependencies:**
```bash
conda activate <your-env-name>
conda env update -f environment.yml
```

**Create `.env` file:**
```bash
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Training the Router Model

```bash
python router_model.py
```

**Note:** If you want to train with full training dataset, you can modify lines 147-148 in router_model.py. If you don't have access to the training data, you can keep it commented to use test data instead.

```python
# Uncomment these lines if using training data:
# line 147
# line 148
```

### Building the Vector Database

```bash
python vector_db_builder.py
```

This creates vector database records for efficient similarity search.

### Running the Application

```bash
streamlit run app.py
```

The application will launch in your default web browser.

## Project Structure

```
.
├── app.py                              # Main Streamlit application
├── bird_db_reader.py                   # Database schema reader
├── bird_evaluator.py                   # Evaluation metrics
├── vector_db_builder.py                # Vector database construction
├── reprocess_results.py                # SQL compiler testing (offline)
├── test_system.py                      # System testing suite
├── environment.yml                     # Conda environment
│
├── data/                              # Bird-Bench datasets
│   ├── dev_20240627/                  # Development set
│   │   ├── dev_databases/            # SQLite databases (11 domains)
│   │   │   ├── california_schools/
│   │   │   ├── card_games/
│   │   │   ├── financial/
│   │   │   ├── formula_1/
│   │   │   └── ... (7 more)
│   │   ├── dev.json                  # Query examples
│   │   ├── dev.sql                   # Gold SQL queries
│   │   └── dev_tables.json           # Table schemas
│   └── train/                         # Training set (optional)
│       ├── train_databases.zip
│       ├── train.json
│       └── train_tables.json
│
├── info/                              # Configuration files
│   ├── database_info.json            # Schema metadata for LLM
│   └── database_info_mappings.json   # Vector DB column mappings
│
├── onePassLlmModel/                   # Core pipeline modules
│   ├── bird_pipeline.py              # Main query processing pipeline
│   ├── gpt_ai_engine.py              # GPT-4 integration
│   ├── groq_ai_engine.py             # Groq LLM integration
│   ├── router_model.py               # Query intent classifier
│   ├── router_model_helper.py        # Router utilities
│   ├── sql_compiler.py               # Hybrid SQL generation
│   └── templates.py                  # LLM prompt templates
│
├── my_router_model/                   # Trained router model v1
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── baselines/                         # Baseline comparisons
│   ├── dail_sql/                     # DAIL-SQL baseline
│   │   ├── evaluate_dail_sql.py
│   │   └── results.json
│   └── din_sql/                      # DIN-SQL baseline
│       ├── evaluate_din_sql.py
│       └── results.json
│
├── results/                           # Experiment results
│   ├── pipeline_test_report_gpt.jsonl
│   ├── pipeline_test_report_summary.json
│   └── ... (baseline comparisons)
│
└── extras/                            # Analysis & utilities
    ├── dataset_statistics.py          # Dataset analysis
    ├── financial_analysis.py          # Domain-specific analysis
    ├── collection_viewer.py           # Vector DB viewer
    ├── vector_search_test.py          # Vector search testing
    └── statistics/                    # Visualization outputs
```

## Components

### Pipeline Architecture

The system follows a 5-step hybrid approach:

1. **Intent Classification** (`router_model.py`)
   - Classifies queries as DATABASE_QUERY vs general chat
   - Fine-tuned BERT model on Bird-Bench queries

2. **Query Decomposition** (`templates.py`, LLM)
   - Analyzes natural language query
   - Identifies semantic terms (needs vector search)
   - Identifies exact filters (needs SQL)
   - Plans required table JOINs

3. **Vector Search Execution** (`vector_db_builder.py`)
   - Searches embedded semantic columns (district names, categories, etc.)
   - Returns matching entity IDs with similarity scores
   - Uses sentence transformers for embeddings

4. **Hybrid SQL Generation** (`sql_compiler.py`)
   - Injects vector search results as `WHERE ... IN (...)` clauses
   - Adds exact SQL filters for numeric/categorical conditions
   - Generates efficient JOINs across tables

5. **Execution & Evaluation** (`bird_evaluator.py`)
   - Executes SQL on SQLite databases
   - Compares with gold standard queries
   - Computes execution accuracy metrics

### Core Files

- **`app.py`** - Interactive Streamlit interface for query translation
- **`onePassLlmModel/bird_pipeline.py`** - Main processing pipeline orchestrator
- **`onePassLlmModel/router_model.py`** - Intent classification model trainer
- **`onePassLlmModel/sql_compiler.py`** - Hybrid SQL generation engine
- **`vector_db_builder.py`** - Builds semantic search index using ChromaDB
- **`bird_db_reader.py`** - Reads Bird-Bench database schemas
- **`bird_evaluator.py`** - Evaluates generated SQL against gold standard
- **`test_system.py`** - Automated end-to-end testing
- **`reprocess_results.py`** - Re-evaluates SQL without re-querying LLMs

### Configuration Files

- **`info/database_info.json`** - Complete schema metadata provided to LLM (tables, columns, relationships)
- **`info/database_info_mappings.json`** - Specifies which columns to embed for vector search (semantic vs exact)

### AI Engine Modules

- **`onePassLlmModel/gpt_ai_engine.py`** - OpenAI GPT-4 integration
- **`onePassLlmModel/groq_ai_engine.py`** - Groq Llama integration  
- **`onePassLlmModel/templates.py`** - Prompt engineering templates for query decomposition

## Development Workflow

### Quick Start
1. Set up environment and install dependencies
2. Configure API keys in `.env`
3. Build vector database: `python vector_db_builder.py`
4. Run the app: `streamlit run app.py`

### Full Development Cycle
1. **Train router model** with your dataset (optional)
   ```bash
   python onePassLlmModel/router_model.py
   ```

2. **Build vector database** for semantic search
   ```bash
   python vector_db_builder.py
   ```
   This embeds semantic columns defined in `database_info_mappings.json`

3. **Launch Streamlit app** for interactive queries
   ```bash
   streamlit run app.py
   ```

4. **Run batch evaluation** on test set
   ```bash
   python test_system.py
   ```

5. **Iterate on SQL compilation** without re-querying LLMs
   ```bash
   python reprocess_results.py
   ```

### Baselines Comparison

Compare against state-of-the-art methods:

```bash
# DAIL-SQL baseline
python baselines/dail_sql/evaluate_dail_sql.py

# DIN-SQL baseline  
python baselines/din_sql/evaluate_din_sql.py
```

### Analysis Tools

```bash
# View dataset statistics
python extras/dataset_statistics.py

# Analyze specific domain (e.g., financial)
python extras/financial_analysis.py

# Inspect vector database collections
python extras/collection_viewer.py

# Test vector search functionality
python extras/vector_search_test.py
```

## How It Works

### Example Query
```
"Show me young female clients from Prague with gold cards and loan amounts over 100,000"
```

This query has both:
- **Semantic terms**: "young" (what age?), "Prague" (needs fuzzy matching)
- **Exact terms**: gender=Female, card type=gold, amount>100,000

### Processing Steps

1. **Router classifies** → DATABASE_QUERY

2. **LLM decomposes** query into:
   - Vector searches: "Prague" in district.A2, interpret "young" as age < 30
   - SQL filters: gender='F', card.type='gold', loan.amount>100000
   - JOINs: client → district, client → disposition → account → loan

3. **Vector search** finds: Prague variants → ('Praha - zapad', 'Hl.m. Praha', etc)

4. **SQL generation**:
   ```sql
   SELECT DISTINCT client.*, district.A2, card.type, loan.amount
   FROM client
   JOIN district ON client.district_id = district.district_id
   JOIN disposition ON client.client_id = disposition.client_id
   JOIN account ON disposition.account_id = account.account_id
   JOIN loan ON account.account_id = loan.account_id
   JOIN card ON disposition.disp_id = card.disp_id
   WHERE (dist.A2 = 'Praha - zapad')  -- From vector search
     AND client.gender = 'F'               -- SQL filter
     AND card.type = 'gold'                -- SQL filter
     AND loan.amount > 100000              -- SQL filter
     AND YEAR(NOW()) - YEAR(client.birth_date) < 30  -- Derived from "young"
   ```

5. **Database executes** and returns results

### Why Hybrid?

| Aspect | Pure SQL | Pure Vector | **Hybrid** ✓ |
|--------|----------|-------------|--------------|
| Exact filters (amount > 100k) | ✓ | ✗ | ✓ |
| Semantic search ("Prague") | ✗ | ✓ | ✓ |
| Complex JOINs | ✓ | ✗ | ✓ |
| Handles typos/variations | ✗ | ✓ | ✓ |
| Performance | ✓ | Slow | ✓ |

**Missing training data?**  
Update `router_model.py` lines 147-148 to use test data instead.

**API key errors?**  
Ensure your `.env` file is in the project root and contains valid API keys.

**Import errors?**  
Run `conda env update -f environment.yml` to ensure all dependencies are installed.

## License

Please refer to the Bird-Bench dataset license for data usage terms.