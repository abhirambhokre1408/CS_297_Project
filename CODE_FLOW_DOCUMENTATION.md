# HyperRAG Code Flow Documentation

## Table of Contents
1. [Dataset Structure](#1-dataset-structure)
2. [Step 0: Dataset Preprocessing](#2-step-0-dataset-preprocessing)
3. [Step 1: HyperGraph Construction](#3-step-1-hypergraph-construction)
4. [Step 2: Question Extraction](#4-step-2-question-extraction)
5. [Step 3: Response Generation](#5-step-3-response-generation)

---

## 1. Dataset Structure

### 1.1 Dataset Format

The project uses **JSONL (JSON Lines)** format, where each line is a separate JSON object. The datasets are organized by domain in the `datasets/` directory:

```
datasets/
├── mix/
│   └── mix.jsonl
├── agriculture/
│   └── agriculture.jsonl
├── art/
│   └── art.jsonl
├── fin/
│   └── fin.jsonl
├── legal/
│   └── legal.jsonl
├── mathematics/
│   └── mathematics.jsonl
├── neurology/
│   └── neurology.jsonl
├── pathology/
│   └── pathology.jsonl
└── physics/
    └── physics.jsonl
```

### 1.2 JSON Object Structure

Each line in the JSONL file contains a JSON object with the following structure:

```json
{
    "input": "Question text (optional)",
    "context": "The main text content/document to be processed",
    "dataset": "Source dataset name (e.g., 'narrativeqa')",
    "label": "Dataset label (e.g., 'longbench')",
    "answers": ["Expected answer(s) (optional)"],
    "_id": "Unique identifier for the record",
    "length": 7966
}
```

**Key Fields:**
- **`context`**: The primary field used by HyperRAG. Contains the actual text/document content that will be processed for entity extraction and knowledge graph construction.
- **`input`**: Optional question/query text (used for evaluation purposes).
- **`answers`**: Optional ground truth answers (used for evaluation).
- **`dataset`**: Source dataset identifier.
- **`_id`**: Unique record identifier.
- **`length`**: Character length of the context.

### 1.3 Dataset Characteristics

- **Format**: JSONL (one JSON object per line)
- **Encoding**: UTF-8
- **Content**: Each record contains a `context` field with substantial text (ranging from hundreds to thousands of characters)
- **Domains**: Multiple domains including literature (narrativeqa), finance, legal, mathematics, science, etc.

---

## 2. Step 0: Dataset Preprocessing

**File**: `reproduce/Step_0.py`  
**Function**: `extract_unique_contexts(input_directory, output_directory)`

### 2.1 Purpose

Step 0 performs **deduplication** of contexts from the JSONL dataset files. It extracts unique context strings and saves them to a JSON file for use in subsequent steps.

### 2.2 Process Flow

#### 2.2.1 Input
- **Directory**: `datasets/{data_name}/` containing `.jsonl` files
- **Example**: `datasets/mix/mix.jsonl`

#### 2.2.2 Processing Steps

1. **File Discovery** (Line 10):
   ```python
   jsonl_files = list(in_dir.glob("*.jsonl"))
   ```
   - Finds all `.jsonl` files in the input directory

2. **Line-by-Line Processing** (Lines 23-36):
   ```python
   for line_number, line in enumerate(infile, start=1):
       json_obj = json.loads(line)
       context = json_obj.get("context")
       if context and context not in unique_contexts_dict:
           unique_contexts_dict[context] = None
   ```
   - Reads each line as a separate JSON object
   - Extracts the `context` field
   - Uses dictionary keys for automatic deduplication (dictionary keys are unique)
   - Only adds contexts that haven't been seen before

3. **Error Handling**:
   - Catches `JSONDecodeError` for malformed JSON lines
   - Handles file I/O errors gracefully
   - Continues processing even if individual lines fail

#### 2.2.3 Output
- **File**: `caches/{data_name}/contexts/{data_name}_unique_contexts.json`
- **Format**: JSON array of unique context strings
- **Example**: `caches/mix/contexts/mix_unique_contexts.json`

**Output Structure:**
```json
[
    "Context string 1...",
    "Context string 2...",
    "Context string 3...",
    ...
]
```

### 2.3 Key Features

- **Deduplication**: Ensures each unique context appears only once
- **Automatic Directory Creation**: Creates output directory if it doesn't exist
- **Skip Existing Files**: If output file already exists, skips processing (line 15-16)
- **Progress Reporting**: Prints number of unique contexts found

### 2.4 Example Usage

```bash
python reproduce/Step_0.py -i datasets/mix -o caches/mix/contexts
```

**Output Example:**
```
Found 1 JSONL files.
Processing file: mix.jsonl
There are 1234 unique `context` entries in the file mix.jsonl.
Unique `context` entries have been saved to: mix_unique_contexts.json
All files have been processed.
```

---

## 3. Step 1: HyperGraph Construction

**File**: `reproduce/Step_1.py`  
**Main Entry Point**: `HyperRAG.insert()` → `HyperRAG.ainsert()`  
**Core Operations**: Defined in `hyperrag/hyperrag.py` and `hyperrag/operate.py`

### 3.1 Purpose

Step 1 builds the HyperGraph knowledge base by:
1. Loading unique contexts from Step 0
2. Chunking the text into manageable pieces
3. Extracting entities and relationships using LLM
4. Constructing a hypergraph structure
5. Storing everything in vector databases and graph storage

### 3.2 HyperRAG Class Initialization

**File**: `hyperrag/hyperrag.py`  
**Function**: `HyperRAG.__post_init__()`

#### 3.2.1 Storage Components Initialized

When `HyperRAG` is instantiated, it creates multiple storage components:

1. **Full Documents Storage** (`JsonKVStorage`):
   - Namespace: `"full_docs"`
   - File: `kv_store_full_docs.json`
   - Stores: Complete original documents with hash-based IDs

2. **Text Chunks Storage** (`JsonKVStorage`):
   - Namespace: `"text_chunks"`
   - File: `kv_store_text_chunks.json`
   - Stores: Processed text chunks with metadata

3. **HyperGraph Storage** (`HypergraphStorage`):
   - Namespace: `"chunk_entity_relation"`
   - File: `hypergraph_chunk_entity_relation.hgdb`
   - Stores: Graph structure with vertices (entities) and hyperedges (relationships)

4. **Vector Databases** (`NanoVectorDBStorage`):
   - **Entities VDB**: `vdb_entities.json` - Semantic search for entities
   - **Relationships VDB**: `vdb_relationships.json` - Semantic search for relationships
   - **Chunks VDB**: `vdb_chunks.json` - Semantic search for text chunks

5. **LLM Response Cache** (`JsonKVStorage`, optional):
   - Namespace: `"llm_response_cache"`
   - File: `kv_store_llm_response_cache.json`
   - Caches LLM responses to avoid redundant API calls

### 3.3 Document Insertion Process

**Function**: `HyperRAG.ainsert(string_or_strings)` in `hyperrag/hyperrag.py`

#### 3.3.1 Phase 1: Document Processing (Lines 178-186)

```python
new_docs = {
    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
    for c in string_or_strings
}
_add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
```

**Process:**
- Converts input strings to documents with hash-based IDs (format: `doc-{hash}`)
- Filters out documents that already exist in storage
- Only processes new documents

**Function Used**: `compute_mdhash_id()` from `hyperrag/utils.py`

#### 3.3.2 Phase 2: Text Chunking (Lines 190-214)

**Function**: `chunking_by_token_size()` in `hyperrag/operate.py` (lines 34-52)

```python
chunks = {
    compute_mdhash_id(dp["content"], prefix="chunk-"): {
        **dp,
        "full_doc_id": doc_key,
    }
    for dp in chunking_by_token_size(
        doc["content"],
        overlap_token_size=self.chunk_overlap_token_size,
        max_token_size=self.chunk_token_size,
        tiktoken_model=self.tiktoken_model_name,
    )
}
```

**Chunking Parameters** (from `HyperRAG.__init__`):
- `chunk_token_size`: 1200 tokens (default)
- `chunk_overlap_token_size`: 100 tokens (default)
- `tiktoken_model`: "gpt-4o-mini" (default)

**Process:**
1. Tokenizes text using `tiktoken` library
2. Splits into overlapping chunks of max 1200 tokens
3. Each chunk has 100 token overlap with previous chunk
4. Creates chunk metadata:
   - `tokens`: Number of tokens
   - `content`: Chunk text
   - `chunk_order_index`: Position in document
   - `full_doc_id`: Link to source document

**Function Used**: `chunking_by_token_size()` from `hyperrag/operate.py`

#### 3.3.3 Phase 3: Entity and Relationship Extraction (Lines 219-226)

**Function**: `extract_entities()` in `hyperrag/operate.py` (lines 402-616)

This is the **core extraction process** that builds the knowledge graph.

##### 3.3.3.1 LLM-Based Extraction

For each chunk, the system:

1. **Initial Extraction Prompt** (Lines 414-431):
   - Uses `PROMPTS["entity_extraction"]` from `hyperrag/prompt.py`
   - Prompts LLM to extract:
     - **Entities**: Name, type, description, additional properties
     - **Low-order Hyperedges**: 2-entity relationships
     - **High-level keywords**: Main themes/concepts
     - **High-order Hyperedges**: 3+ entity relationships

2. **Iterative Gleaning** (Lines 452-468):
   ```python
   for now_glean_index in range(entity_extract_max_gleaning):
       glean_result = await use_llm_func(continue_prompt, history_messages=history)
       # ... check if more extraction needed
   ```
   - Uses `PROMPTS["entity_continue_extraction"]` to find missed entities
   - Uses `PROMPTS["entity_if_loop_extraction"]` to check if more extraction needed
   - Default: 1 iteration (`entity_extract_max_gleaning = 1`)

3. **Response Parsing** (Lines 470-479):
   - Splits response by delimiters: `record_delimiter` ("\n") and `completion_delimiter` ("<|COMPLETE|>")
   - Extracts tuples using regex: `re.search(r"\((.*)\)", record)`

##### 3.3.3.2 Entity Parsing

**Function**: `_handle_single_entity_extraction()` (lines 174-195)

**Format**: `("Entity" | <name> | <type> | <description> | <additional_properties>)`

**Extracted Fields:**
- `entity_name`: Uppercased, cleaned string
- `entity_type`: One of ["organization", "person", "geo", "event", "role", "concept"]
- `description`: Comprehensive entity description
- `additional_properties`: Time, space, emotion, motivation, etc.
- `source_id`: Chunk ID where entity was found

##### 3.3.3.3 Relationship Parsing

**Low-order Hyperedges** (2 entities):
- **Function**: `_handle_single_relationship_extraction_low()` (lines 198-223)
- **Format**: `("Low-order Hyperedge" | <entity1> | <entity2> | <description> | <keywords> | <weight>)`
- **Fields**: Description, keywords, weight (0-10, default 0.75)

**High-order Hyperedges** (3+ entities):
- **Function**: `_handle_single_relationship_extraction_high()` (lines 225-249)
- **Format**: `("High-order Hyperedge" | <entity1> | ... | <entityN> | <description> | <generalization> | <keywords> | <weight>)`
- **Fields**: Description, generalization, keywords, weight

##### 3.3.3.4 Entity Merging

**Function**: `_merge_nodes_then_upsert()` (lines 252-330)

For each unique entity:

1. **Check Existing** (Line 263):
   ```python
   already_node = await knowledge_hypergraph_inst.get_vertex(entity_name)
   ```

2. **Merge Information**:
   - `entity_type`: Most common type (using Counter)
   - `description`: Combines all descriptions (separated by `GRAPH_FIELD_SEP` = "<SEP>")
   - `additional_properties`: Combines all properties
   - `source_id`: Combines all source chunk IDs

3. **Summarization** (if too long):
   - `_handle_entity_summary()`: If description > 500 tokens
   - `_handle_entity_additional_properties()`: If properties > 250 tokens
   - Uses LLM to create concise summaries

4. **Upsert to Hypergraph**:
   ```python
   await knowledge_hypergraph_inst.upsert_vertex(entity_name, entity_data)
   ```

##### 3.3.3.5 Relationship Merging

**Function**: `_merge_edges_then_upsert()` (lines 331-399)

For each unique entity set (tuple of entity names):

1. **Merge Information**:
   - `description`: Combines all relationship descriptions
   - `keywords`: Combines and deduplicates keywords
   - `weight`: Averages weights
   - `source_id`: Combines all source chunk IDs

2. **Summarization** (if too long):
   - `_handle_relation_summary()`: If description > 750 tokens
   - `_handle_relation_keywords_summary()`: If keywords > 100 tokens

3. **Upsert to Hypergraph**:
   ```python
   await knowledge_hypergraph_inst.upsert_hyperedge(id_set, edge_data)
   ```

##### 3.3.3.6 Vector Database Upsertion (Lines 594-614)

**Entities Vector DB**:
```python
data_for_vdb = {
    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        "content": dp["entity_name"] + dp["description"],
        "entity_name": dp["entity_name"],
    }
    for dp in all_entities_data
}
await entity_vdb.upsert(data_for_vdb)
```
- Creates embeddings for `entity_name + description`
- Stores in `vdb_entities.json`

**Relationships Vector DB**:
```python
data_for_vdb = {
    compute_mdhash_id(str(sorted(dp["id_set"])), prefix="rel-"): {
        "id_set": dp["id_set"],
        "content": dp["keywords"] + str(dp["id_set"]) + dp["description"],
    }
    for dp in all_relationships_data
}
await relationships_vdb.upsert(data_for_vdb)
```
- Creates embeddings for `keywords + id_set + description`
- Stores in `vdb_relationships.json`

**Function Used**: `embedding_func()` - Creates vector embeddings using OpenAI API

### 3.4 Output Files Generated

After Step 1 completes, the following files are created in `caches/{data_name}/`:

1. **`kv_store_full_docs.json`**: Complete documents
2. **`kv_store_text_chunks.json`**: Text chunks with metadata
3. **`hypergraph_chunk_entity_relation.hgdb`**: Hypergraph structure
4. **`vdb_entities.json`**: Entity vector database
5. **`vdb_relationships.json`**: Relationship vector database
6. **`vdb_chunks.json`**: Chunk vector database
7. **`kv_store_llm_response_cache.json`**: LLM response cache (optional)
8. **`HyperRAG.log`**: Processing logs

### 3.5 HyperRAG Methods Used

From `hyperrag/hyperrag.py`:
- `HyperRAG.__init__()`: Initializes storage components
- `HyperRAG.insert()`: Synchronous wrapper (calls `ainsert`)
- `HyperRAG.ainsert()`: Main async insertion method

From `hyperrag/operate.py`:
- `chunking_by_token_size()`: Text chunking
- `extract_entities()`: Entity/relationship extraction
- `_merge_nodes_then_upsert()`: Entity merging
- `_merge_edges_then_upsert()`: Relationship merging

From `hyperrag/storage.py`:
- `JsonKVStorage.upsert()`: Key-value storage
- `NanoVectorDBStorage.upsert()`: Vector database operations
- `HypergraphStorage.upsert_vertex()`: Graph vertex operations
- `HypergraphStorage.upsert_hyperedge()`: Graph hyperedge operations

---

## 4. Step 2: Question Extraction

**File**: `reproduce/Step_2_extract_question.py`  
**Main Execution**: Script in `if __name__ == "__main__"`

### 4.1 Purpose

Step 2 generates evaluation questions from the unique contexts using LLM. It creates questions of varying complexity (1-stage, 2-stage, or 3-stage) that test the knowledge stored in the HyperGraph.

### 4.2 Process Flow

#### 4.2.1 Input Loading (Lines 198-201)

```python
contexts_path = WORKING_DIR / "contexts" / f"{data_name}_unique_contexts.json"
with open(contexts_path, "r", encoding="utf-8") as f:
    unique_contexts = json.load(f)
```

- Loads unique contexts from Step 0 output
- File: `caches/{data_name}/contexts/{data_name}_unique_contexts.json`

#### 4.2.2 Configuration (Lines 203-212)

```python
max_questions = 5              # Number of questions to generate
max_chunks = 3                # Maximum chunks to combine
min_chunks = 1                # Minimum chunks to combine
MAX_TOKENS_FOR_CONTEXT = 5500 # Maximum tokens per context
question_stage = 2            # 1, 2, or 3-stage questions
```

#### 4.2.3 Pre-filtering Valid Combinations (Lines 217-253)

**Problem Solved**: Many contexts exceed token limits, causing retry loops.

**Solution**: Pre-filter all valid context combinations before randomly selecting.

```python
# Try different chunk sizes (3, 2, 1)
for num_chunks in range(max_chunks, min_chunks - 1, -1):
    for idx in range(max_idx):
        block = unique_contexts[idx: idx + num_chunks]
        context = "".join(block)
        token_len = len(encoding.encode(context))
        
        if token_len <= MAX_TOKENS_FOR_CONTEXT:
            valid_combinations.append((idx, num_chunks, context, token_len))
```

**Process:**
1. Tries combining 3, 2, or 1 consecutive contexts
2. Checks token count for each combination
3. Stores valid combinations (≤ 5500 tokens)
4. Falls back to truncation if no valid combinations found

**Function Used**: `truncate_context_to_tokens()` (lines 44-62) - Intelligently truncates while preserving sentence boundaries

#### 4.2.4 Random Selection (Lines 258-260)

```python
selected_indices = np.random.choice(len(valid_combinations), 
                                   size=min(max_questions, len(valid_combinations)), 
                                   replace=False)
selected_combinations = [valid_combinations[i] for i in selected_indices]
```

- Randomly samples from valid combinations
- Ensures no duplicates (`replace=False`)

#### 4.2.5 Question Generation (Lines 262-286)

For each selected combination:

1. **Prompt Formatting** (Line 265):
   ```python
   prompt = question_prompt[question_stage].format(context=context)
   ```
   - Uses prompt templates from `question_prompt` dictionary (lines 65-140)
   - Different prompts for 1-stage, 2-stage, and 3-stage questions

2. **LLM Call** (Line 267):
   ```python
   resp = llm_model_func(prompt)
   ```
   - **Function**: `llm_model_func()` (lines 29-41)
   - Uses OpenAI API with model `gpt-4o` (configurable)
   - Client reuse for efficiency (lines 20-27)

3. **Question Extraction** (Line 275):
   ```python
   m = re.findall(r'"Question":\s*"([^"]+)"', resp)
   ```
   - Extracts question from JSON-formatted LLM response
   - Regex pattern matches JSON structure

4. **Error Handling**:
   - Catches LLM API errors
   - Skips if question extraction fails
   - Continues to next combination

#### 4.2.6 Question Types

**1-Stage Questions** (Simple, focused):
- Single question
- No conjunctions ("and", "or", "specifically")
- Tests one aspect/detail

**2-Stage Questions** (Progressive):
- Two interconnected sub-questions
- Connected by transitional phrases ("and", "specifically")
- Example: "What is X and specifically how does Y work?"

**3-Stage Questions** (Complex):
- Three interconnected sub-questions
- Demonstrates progressive relationship
- Most challenging type

#### 4.2.7 Output Files (Lines 288-302)

**Questions File**:
- Path: `caches/{data_name}/questions/{question_stage}_stage.json`
- Format: JSON array of question strings
- Example: `["Question 1", "Question 2", ...]`

**References File**:
- Path: `caches/{data_name}/questions/{question_stage}_stage_ref.json`
- Format: JSON array of reference contexts
- Contains the contexts used to generate each question

### 4.3 Key Functions Used

**From Step_2_extract_question.py:**
- `llm_model_func()`: LLM API calls
- `get_openai_client()`: Client singleton pattern
- `truncate_context_to_tokens()`: Context truncation

**External Libraries:**
- `tiktoken`: Token counting
- `numpy`: Random selection
- `openai.OpenAI`: API client

### 4.4 Example Output

**Questions File** (`2_stage.json`):
```json
[
  "What are the main components of the central nervous system and how do they interact to process sensory information?",
  "How does photosynthesis convert light energy and what role do chlorophyll molecules play in this process?",
  ...
]
```

**References File** (`2_stage_ref.json`):
```json
[
  "Context text 1...",
  "Context text 2...",
  ...
]
```

---

## 5. Step 3: Response Generation

**File**: `reproduce/Step_3_response_question.py`  
**Main Execution**: Script in `if __name__ == "__main__"`  
**Core Query Methods**: Defined in `hyperrag/operate.py` and `hyperrag/hyperrag.py`

### 5.1 Purpose

Step 3 generates answers to the extracted questions using HyperRAG. It supports multiple query modes (naive, hyper, hyper-lite, graph, llm) that use different retrieval strategies.

### 5.2 Process Flow

#### 5.2.1 Input Loading (Lines 88-91)

```python
question_file_path = Path(WORKING_DIR / f"questions/{question_stage}_stage.json")
queries = extract_queries(question_file_path)
```

**Function**: `extract_queries()` (lines 41-44)
- Loads questions from Step 2 output
- Returns list of question strings

#### 5.2.2 HyperRAG Initialization (Lines 93-99)

```python
rag = HyperRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
    ),
)
```

**Components Initialized:**
- Loads existing HyperGraph from Step 1
- Initializes vector databases
- Sets up LLM and embedding functions

#### 5.2.3 Query Mode Configuration (Lines 100-104)

```python
mode = "naive"        # or "hyper", "hyper-lite", "graph", "llm"
query_param = QueryParam(mode=mode)
```

**Available Modes:**
1. **`"naive"`**: Simple vector search on chunks (baseline RAG)
2. **`"hyper"`**: Full HyperRAG (entity + relationship retrieval)
3. **`"hyper-lite"`**: HyperRAG with only entity retrieval
4. **`"graph"`**: Graph traversal-based query
5. **`"llm"`**: Direct LLM (no retrieval)

#### 5.2.4 Query Processing (Lines 47-53, 67-80)

**Function**: `process_query()` (lines 47-53)

```python
async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}
```

**Process:**
1. Calls `rag_instance.aquery()` with query and parameters
2. Returns result or error
3. Handles exceptions gracefully

**Function**: `run_queries_and_save_to_json()` (lines 56-80)

```python
for query_text in tqdm(queries, desc="Processing queries", unit="query"):
    result, error = loop.run_until_complete(
        process_query(query_text, rag_instance, query_param)
    )
    # Save to JSON file
```

- Processes each question sequentially
- Saves results incrementally to JSON file
- Shows progress bar

### 5.3 Query Modes Detailed

#### 5.3.1 Naive Query Mode

**Function**: `naive_query()` in `hyperrag/operate.py` (lines 1581-1628)

**Process:**
1. **Vector Search** (Line 1589):
   ```python
   results = await chunks_vdb.query(query, top_k=query_param.top_k)
   ```
   - Direct semantic search on text chunks
   - Uses vector similarity (cosine similarity)

2. **Retrieve Chunks** (Line 1593):
   ```python
   chunks = await text_chunks_db.get_by_ids(chunks_ids)
   ```

3. **Truncate** (Lines 1595-1599):
   - Limits to `max_token_for_text_unit` tokens (default: 1600)

4. **LLM Response** (Lines 1604-1611):
   - Uses `PROMPTS["naive_rag_response"]`
   - Simple context + query → answer

**Characteristics:**
- No graph structure used
- Fastest mode
- Baseline RAG performance

#### 5.3.2 Hyper Query Mode

**Function**: `hyper_query()` in `hyperrag/operate.py` (lines 1062-1177)

This is the **most comprehensive** query mode, using both entity and relationship retrieval.

##### Phase 1: Keyword Extraction (Lines 1075-1103)

```python
kw_prompt = PROMPTS["keywords_extraction"].format(query=query)
result = await use_model_func(kw_prompt)
keywords_data = json.loads(result)
entity_keywords = keywords_data.get("low_level_keywords", [])
relation_keywords = keywords_data.get("high_level_keywords", [])
```

**Process:**
- Extracts low-level keywords (entities, details)
- Extracts high-level keywords (concepts, themes)
- Uses LLM to analyze query intent

##### Phase 2: Entity-based Context Building (Lines 1109-1143)

**Function**: `_build_entity_query_context()` (lines 619-740)

**Process:**

1. **Entity Retrieval** (Lines 626-638):
   ```python
   results = await entities_vdb.query(entity_keywords, top_k=query_param.top_k)
   node_datas = await knowledge_hypergraph_inst.get_vertex(entity_name)
   ```
   - Vector search on entities
   - Retrieve entity data from hypergraph

2. **Find Related Text Units** (Lines 640-650):
   ```python
   use_text_units = await _find_most_related_text_unit_from_entities(
       node_datas, query_param, text_chunks_db, knowledge_hypergraph_inst
   )
   ```
   - Gets source chunks from entities
   - Gets neighbor hyperedges
   - Scores chunks by relationship counts
   - **Function**: `_find_most_related_text_unit_from_entities()` (lines 744-822)

3. **Find Related Relationships** (Lines 652-656):
   ```python
   use_relations = await _find_most_related_edges_from_entities(
       node_datas, query_param, knowledge_hypergraph_inst
   )
   ```
   - Gets hyperedges connected to entities
   - Sorts by degree and weight
   - **Function**: `_find_most_related_edges_from_entities()` (lines 825-859)

4. **Build Context String** (Lines 658-705):
   - Formats as CSV tables:
     - Entities: `[id, entity, type, description, additional properties, rank]`
     - Relationships: `[id, entity set, description, keywords, weight, rank]`
     - Sources: `[id, content]`

##### Phase 3: Relationship-based Context Building (Lines 1145-1171)

**Function**: `_build_relation_query_context()` (lines 862-990)

**Process:**

1. **Relationship Retrieval** (Line 870):
   ```python
   results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
   ```

2. **Find Related Entities** (Lines 899-901):
   ```python
   use_entities = await _find_most_related_entities_from_relationships(
       edge_datas, query_param, knowledge_hypergraph_inst
   )
   ```
   - Extracts entities from hyperedge `id_set`
   - **Function**: `_find_most_related_entities_from_relationships()` (lines 992-1022)

3. **Find Related Text Units** (Lines 902-904):
   ```python
   use_text_units = await _find_related_text_unit_from_relationships(
       edge_datas, query_param, text_chunks_db, knowledge_hypergraph_inst
   )
   ```
   - Gets source chunks from hyperedges
   - **Function**: `_find_related_text_unit_from_relationships()` (lines 1025-1059)

4. **Build Context String**: Same CSV format as entity context

##### Phase 4: Combine Contexts (Lines 1153-1171)

```python
if entity_context and relation_context:
    # Combine both contexts
elif entity_context:
    # Use entity context only
elif relation_context:
    # Use relation context only
```

##### Phase 5: Generate Response (Lines 1173-1177)

```python
sys_prompt = PROMPTS["rag_response"].format(
    context_data=context_string, response_type=query_param.response_type
)
response = await use_model_func(query, system_prompt=sys_prompt)
```

**Characteristics:**
- Uses both entity-level and relationship-level retrieval
- Most comprehensive context
- Highest accuracy (typically)

#### 5.3.3 Hyper Query Lite Mode

**Function**: `hyper_query_lite()` in `hyperrag/operate.py` (lines 1180-1236)

**Process:**
- Similar to `hyper_query()` but **only uses entity-based retrieval**
- No relationship-based context building
- Faster than full hyper mode
- Less comprehensive context

#### 5.3.4 Graph Query Mode

**Function**: `graph_query()` in `hyperrag/operate.py` (lines 1275-1488)

**Process:**
- Similar structure to `hyper_query()`
- More focused on graph traversal
- Uses graph structure more extensively

#### 5.3.5 LLM Query Mode

**Function**: `llm_query()` in `hyperrag/operate.py` (lines 1631-1650)

**Process:**
- Direct LLM call with no retrieval
- No context building
- Pure LLM response (no RAG)

### 5.4 HyperRAG Methods Used

**From `hyperrag/hyperrag.py`:**
- `HyperRAG.__init__()`: Initialization
- `HyperRAG.aquery()`: Main async query method (lines 257-305)
  - Routes to appropriate query function based on mode

**From `hyperrag/operate.py`:**
- `naive_query()`: Naive RAG
- `hyper_query()`: Full HyperRAG
- `hyper_query_lite()`: Lite HyperRAG
- `graph_query()`: Graph-based query
- `llm_query()`: Direct LLM
- `_build_entity_query_context()`: Entity context building
- `_build_relation_query_context()`: Relationship context building
- `_find_most_related_text_unit_from_entities()`: Text unit retrieval
- `_find_most_related_edges_from_entities()`: Relationship retrieval
- `_find_most_related_entities_from_relationships()`: Entity retrieval from relationships
- `_find_related_text_unit_from_relationships()`: Text unit retrieval from relationships

**From `hyperrag/storage.py`:**
- `NanoVectorDBStorage.query()`: Vector similarity search
- `HypergraphStorage.get_vertex()`: Get entity data
- `HypergraphStorage.get_hyperedge()`: Get relationship data
- `HypergraphStorage.get_nbr_e_of_vertex()`: Get connected hyperedges
- `HypergraphStorage.get_nbr_v_of_hyperedge()`: Get connected entities

### 5.5 Output Files

**Results File**:
- Path: `caches/{data_name}/response/{mode}_{question_stage}_stage_result.json`
- Format: JSON array of query-result pairs
- Example: `caches/mix/response/hyper_2_stage_result.json`

**Errors File**:
- Path: `caches/{data_name}/response/{mode}_{question_stage}_stage_errors.json`
- Format: JSON array of error records
- Contains queries that failed

**Output Structure:**
```json
[
  {
    "query": "Question text...",
    "result": "Answer text..."
  },
  {
    "query": "Question text...",
    "result": "Answer text..."
  },
  ...
]
```

### 5.6 Example Usage

```python
# In Step_3_response_question.py
mode = "hyper"  # Change this to switch modes
query_param = QueryParam(mode=mode)
```

**Output Files Generated:**
- `hyper_2_stage_result.json`: Results
- `hyper_2_stage_errors.json`: Errors (if any)

---

## Summary: Complete Pipeline Flow

```
1. Dataset (JSONL)
   ↓
2. Step 0: Extract Unique Contexts
   → Output: caches/{data_name}/contexts/{data_name}_unique_contexts.json
   ↓
3. Step 1: Build HyperGraph
   → HyperRAG.insert() → chunking → extract_entities() → build hypergraph
   → Output: hypergraph.hgdb, vdb_*.json, kv_store_*.json
   ↓
4. Step 2: Extract Questions
   → Load contexts → Pre-filter → LLM generation → Extract questions
   → Output: questions/{stage}_stage.json, questions/{stage}_stage_ref.json
   ↓
5. Step 3: Generate Responses
   → Load questions → HyperRAG.aquery() → Query mode routing → Context building → LLM response
   → Output: response/{mode}_{stage}_stage_result.json
```

---

## Key Design Patterns

1. **Async/Await**: All I/O operations use async for efficiency
2. **Singleton Pattern**: OpenAI client reuse (Step 2)
3. **Factory Pattern**: Storage component creation (HyperRAG)
4. **Strategy Pattern**: Multiple query modes (Step 3)
5. **Retry Logic**: Error handling with retries (Step 1, Step 2)
6. **Caching**: LLM response caching to avoid redundant API calls
7. **Modular Design**: Separate functions for each operation
8. **Error Resilience**: Graceful error handling throughout

---

This documentation provides a complete overview of the HyperRAG code flow, from dataset preprocessing through question extraction and response generation.

