# HyperRAG Architecture - Complete Overview

## Table of Contents
1. [Overall Pipeline](#overall-pipeline)
2. [Step 0: Dataset Preprocessing](#step-0-dataset-preprocessing)
3. [Step 1: HyperGraph Construction](#step-1-hypergraph-construction)
4. [Step 2: Question Extraction](#step-2-question-extraction)
5. [Step 3: Question Response Generation](#step-3-question-response-generation)
6. [Core Components](#core-components)
7. [Query Modes](#query-modes)

---

## Overall Pipeline

```
Dataset (JSONL) 
  → Step 0: Extract Unique Contexts
  → Step 1: Build HyperGraph (Entity/Relationship Extraction)
  → Step 2: Extract Questions from Contexts
  → Step 3: Generate Answers using HyperRAG
```

---

## Step 0: Dataset Preprocessing

**File**: `reproduce/Step_0.py`  
**Function**: `extract_unique_contexts(input_directory, output_directory)`

### Process:
1. **Read JSONL files** from `datasets/{data_name}/` directory
2. **Extract unique contexts**: For each line in JSONL:
   - Parse JSON: `json_obj = json.loads(line)`
   - Extract `context` field: `context = json_obj.get("context")`
   - Use dictionary to deduplicate: `unique_contexts_dict[context] = None`
3. **Save to JSON**: Write unique contexts list to `caches/{data_name}/contexts/{data_name}_unique_contexts.json`

### Output:
- List of unique context strings (no duplicates)

---

## Step 1: HyperGraph Construction

**File**: `reproduce/Step_1.py` → `hyperrag/hyperrag.py` → `hyperrag/operate.py`  
**Main Entry Point**: `HyperRAG.insert()` → `HyperRAG.ainsert()`  
**Core Function**: `extract_entities()` in `operate.py`

### 1.1 Initialization (`hyperrag/hyperrag.py`)

**Function**: `HyperRAG.__post_init__()`

**Storage Initialization:**
- `full_docs`: `JsonKVStorage` (namespace="full_docs") - Stores full documents
- `text_chunks`: `JsonKVStorage` (namespace="text_chunks") - Stores text chunks
- `chunk_entity_relation_hypergraph`: `HypergraphStorage` - Stores hypergraph structure
- `entities_vdb`: `NanoVectorDBStorage` (namespace="entities") - Vector DB for entities
- `relationships_vdb`: `NanoVectorDBStorage` (namespace="relationships") - Vector DB for relationships
- `chunks_vdb`: `NanoVectorDBStorage` (namespace="chunks") - Vector DB for text chunks
- `llm_response_cache`: `JsonKVStorage` (optional) - Caches LLM responses

### 1.2 Document Insertion (`hyperrag/hyperrag.py`)

**Function**: `async def ainsert(string_or_strings)`

#### Phase 1: Document Processing
1. **Hash documents**: `compute_mdhash_id(content, prefix="doc-")`
2. **Filter existing**: Only process new documents not in storage
3. **Store full docs**: `new_docs[doc_id] = {"content": content}`

#### Phase 2: Text Chunking (`operate.py`)

**Function**: `chunking_by_token_size(content, overlap_token_size, max_token_size, tiktoken_model)`

**Process:**
- Tokenize text using `tiktoken` (default model: "gpt-4o-mini")
- Split into chunks of size `chunk_token_size` (default: 1200 tokens)
- Overlap between chunks: `chunk_overlap_token_size` (default: 100 tokens)
- Each chunk stores:
  - `tokens`: Number of tokens
  - `content`: Chunk text
  - `chunk_order_index`: Position in document
  - `full_doc_id`: Link to source document

#### Phase 3: Entity and Relationship Extraction (`operate.py`)

**Function**: `async def extract_entities(chunks, knowledge_hypergraph_inst, entity_vdb, relationships_vdb, global_config)`

**Process:**

**3.1 LLM-Based Extraction** (Function: `_process_single_content()`)

For each chunk:
1. **Initial extraction prompt**: Uses `PROMPTS["entity_extraction"]` from `prompt.py`
   - Prompts LLM to extract:
     - **Entities**: Name, type, description, additional properties
     - **Low-order Hyperedges**: Pairs of related entities (2-entity relationships)
     - **High-level keywords**: Main themes/concepts
     - **High-order Hyperedges**: Multi-entity relationships (3+ entities)

2. **Iterative gleaning** (`entity_extract_max_gleaning` times, default: 1):
   - Uses `PROMPTS["entity_continue_extraction"]` to find missed entities
   - Uses `PROMPTS["entity_if_loop_extraction"]` to check if more extraction needed

3. **Parse LLM response**:
   - Split by delimiters: `record_delimiter` ("\n") and `completion_delimiter` ("<|COMPLETE|>")
   - Extract tuples using regex: `re.search(r"\((.*)\)", record)`

**3.2 Entity Parsing** (Function: `_handle_single_entity_extraction()`)

Format: `("Entity" | <name> | <type> | <description> | <additional_properties>)`

Extracts:
- `entity_name`: Uppercased, cleaned
- `entity_type`: One of ["organization", "person", "geo", "event", "role", "concept"]
- `description`: Entity description
- `additional_properties`: Time, space, emotion, motivation, etc.
- `source_id`: Chunk ID where entity was found

**3.3 Relationship Parsing**

**Low-order Hyperedges** (Function: `_handle_single_relationship_extraction_low()`)

Format: `("Low-order Hyperedge" | <entity1> | <entity2> | <description> | <keywords> | <weight>)`

- 2 entities (pair relationship)
- `description`: Why entities are related
- `keywords`: Summary keywords
- `weight`: Relationship strength (0-10, default: 0.75)

**High-order Hyperedges** (Function: `_handle_single_relationship_extraction_high()`)

Format: `("High-order Hyperedge" | <entity1> | <entity2> | ... | <entityN> | <description> | <generalization> | <keywords> | <weight>)`

- 3+ entities (multi-entity relationship)
- `description`: Comprehensive relationship description
- `generalization`: Concise summary
- `keywords`: Theme keywords
- `weight`: Association strength

**3.4 Entity Merging** (Function: `_merge_nodes_then_upsert()`)

For each unique entity:
1. **Check if exists** in hypergraph: `knowledge_hypergraph_inst.get_vertex(entity_name)`
2. **Merge information**:
   - `entity_type`: Most common type (Counter-based)
   - `description`: Combine all descriptions (separated by `GRAPH_FIELD_SEP`)
   - `additional_properties`: Combine all properties
   - `source_id`: Combine all source chunk IDs
3. **Summarize if too long**:
   - `_handle_entity_summary()`: If description > 500 tokens
   - `_handle_entity_additional_properties()`: If properties > 250 tokens
4. **Upsert to hypergraph**: `knowledge_hypergraph_inst.upsert_vertex(entity_name, entity_data)`

**3.5 Relationship Merging** (Function: `_merge_edges_then_upsert()`)

For each unique entity set (tuple of entity names):
1. **Check if exists**: `knowledge_hypergraph_inst.get_hyperedge(id_set)`
2. **Merge information**:
   - `description`: Combine all relationship descriptions
   - `keywords`: Combine and deduplicate keywords
   - `weight`: Average weight
   - `source_id`: Combine all source chunk IDs
3. **Summarize if too long**:
   - `_handle_relation_summary()`: If description > 750 tokens
   - `_handle_relation_keywords_summary()`: If keywords > 100 tokens
4. **Upsert to hypergraph**: `knowledge_hypergraph_inst.upsert_hyperedge(id_set, edge_data)`

**3.6 Vector Database Upsertion**

**Entities** (`entity_vdb.upsert()`):
- Content: `entity_name + description`
- Metadata: `entity_name`
- Creates embeddings using `embedding_func`

**Relationships** (`relationships_vdb.upsert()`):
- Content: `keywords + id_set + description`
- Metadata: `id_set` (tuple of entity names)
- Creates embeddings using `embedding_func`

**3.7 Hypergraph Storage**

**Storage Backend**: `HypergraphStorage` (wraps `HypergraphDB` from `hyperdb` library)

**File**: `caches/{data_name}/hypergraph_chunk_entity_relation.hgdb`

**Structure**:
- **Vertices**: Entities (each entity_name is a vertex)
- **Hyperedges**: Relationships (each entity set is a hyperedge)
- **Vertex data**: `{entity_name, entity_type, description, additional_properties, source_id}`
- **Hyperedge data**: `{id_set, description, keywords, weight, source_id, level_hg}`

### 1.3 Storage Persistence

**Function**: `async def _insert_done()`

All storage instances call `index_done_callback()`:
- `JsonKVStorage`: Writes to JSON files
- `NanoVectorDBStorage`: Saves vector DB to JSON
- `HypergraphStorage`: Saves hypergraph to `.hgdb` file

---

## Step 2: Question Extraction

**File**: `reproduce/Step_2_extract_question.py`  
**Main Function**: Script execution in `if __name__ == "__main__"`

### Process:

1. **Load contexts**: Read `caches/{data_name}/contexts/{data_name}_unique_contexts.json`

2. **Pre-filter valid combinations**:
   - Try chunk sizes: 3, 2, 1 (in order)
   - For each chunk size, check token count ≤ 5500 tokens
   - Store valid combinations: `(idx, num_chunks, context, token_len)`

3. **Random selection**: Select `max_questions` (default: 5) from valid combinations

4. **LLM-based question generation**:
   - **Prompt**: `question_prompt[question_stage]` (1, 2, or 3-stage questions)
   - **Function**: `llm_model_func(prompt)` → Calls OpenAI API
   - **Extract question**: Regex `r'"Question":\s*"([^"]+)"'`

5. **Save outputs**:
   - Questions: `caches/{data_name}/questions/{stage}_stage.json`
   - References: `caches/{data_name}/questions/{stage}_stage_ref.json`

### Question Types:

- **1-stage**: Single, focused question (no conjunctions)
- **2-stage**: Question with 2 interconnected sub-questions (e.g., "What is X and specifically how does Y work?")
- **3-stage**: Question with 3 interconnected sub-questions

---

## Step 3: Question Response Generation

**File**: `reproduce/Step_3_response_question.py` → `hyperrag/hyperrag.py`  
**Main Function**: `HyperRAG.aquery()` → Various query modes

### Query Modes:

#### 1. Naive Query (`naive_query()`)

**Function**: `async def naive_query(query, chunks_vdb, text_chunks_db, query_param, global_config)`

**Process**:
1. **Vector search**: `chunks_vdb.query(query, top_k=query_param.top_k)` - Direct semantic search on chunks
2. **Retrieve chunks**: Get chunk content from `text_chunks_db`
3. **Truncate**: Limit to `max_token_for_text_unit` tokens
4. **LLM response**: Use `PROMPTS["naive_rag_response"]` with retrieved chunks
5. **Return**: Final answer

**Characteristics**: Simple RAG - no graph structure used

---

#### 2. Hyper Query (`hyper_query()`)

**Function**: `async def hyper_query(query, knowledge_hypergraph_inst, entities_vdb, relationships_vdb, text_chunks_db, query_param, global_config)`

**Process**:

**Phase 1: Keyword Extraction**
- **Function**: `PROMPTS["keywords_extraction"]`
- **Output**: 
  - `low_level_keywords`: Specific entities/details
  - `high_level_keywords`: Overarching concepts/themes

**Phase 2: Entity-based Context Building** (if `entity_keywords` exist)

**Function**: `_build_entity_query_context()`

1. **Entity retrieval**:
   - `entities_vdb.query(entity_keywords, top_k=query_param.top_k)` - Vector search
   - Get entities from hypergraph: `knowledge_hypergraph_inst.get_vertex(entity_name)`

2. **Find related text units**:
   - **Function**: `_find_most_related_text_unit_from_entities()`
   - Get source chunks from entities: `split_string_by_multi_markers(entity["source_id"])`
   - Get neighbor hyperedges: `knowledge_hypergraph_inst.get_nbr_e_of_vertex(entity_name)`
   - Score chunks by relation counts (how many relationships connect to them)
   - Truncate to `max_token_for_text_unit` tokens

3. **Find related relationships**:
   - **Function**: `_find_most_related_edges_from_entities()`
   - Get hyperedges connected to entities
   - Sort by hyperedge degree and weight
   - Truncate to `max_token_for_relation_context` tokens

4. **Build context string**:
   - Format as CSV tables:
     - Entities table: `[id, entity, type, description, additional properties, rank]`
     - Relationships table: `[id, entity set, description, keywords, weight, rank]`
     - Sources table: `[id, content]`

**Phase 3: Relationship-based Context Building** (if `relation_keywords` exist)

**Function**: `_build_relation_query_context()`

1. **Relationship retrieval**:
   - `relationships_vdb.query(relation_keywords, top_k=query_param.top_k)` - Vector search
   - Get hyperedges from hypergraph

2. **Find related entities**:
   - **Function**: `_find_most_related_entities_from_relationships()`
   - Extract entities from hyperedge `id_set`
   - Get entity data from hypergraph
   - Sort by vertex degree
   - Truncate to `max_token_for_entity_context` tokens

3. **Find related text units**:
   - **Function**: `_find_related_text_unit_from_relationships()`
   - Get source chunks from hyperedges
   - Sort by order
   - Truncate to `max_token_for_text_unit` tokens

4. **Build context string**: Same CSV format as entity context

**Phase 4: Combine Contexts**

- If both entity and relation contexts exist: Combine them
- If only one exists: Use that one

**Phase 5: Generate Response**

- **System prompt**: `PROMPTS["rag_response"]` with context data
- **Additional prompt**: `PROMPTS["rag_define"]` with keywords if both contexts exist
- **LLM call**: `use_model_func(query, system_prompt=sys_prompt)`
- **Return**: Final answer

**Characteristics**: Uses both entity-level and relationship-level retrieval

---

#### 3. Hyper Query Lite (`hyper_query_lite()`)

**Function**: `async def hyper_query_lite(query, knowledge_hypergraph_inst, entities_vdb, text_chunks_db, query_param, global_config)`

**Process**:
- Similar to `hyper_query()` but **only uses entity-based retrieval**
- No relationship-based context building
- Faster but less comprehensive

---

#### 4. Graph Query (`graph_query()`)

**Function**: `async def graph_query(query, knowledge_hypergraph_inst, entities_vdb, relationships_vdb, text_chunks_db, query_param, global_config)`

**Process**:
- Similar structure to `hyper_query()` but uses graph traversal
- More focused on graph structure relationships

---

#### 5. LLM Query (`llm_query()`)

**Function**: `async def llm_query(query, query_param, global_config)`

**Process**:
- Direct LLM call with no retrieval
- No context building
- Pure LLM response

---

## Core Components

### Storage Abstractions (`hyperrag/storage.py`)

#### 1. JsonKVStorage
- **Class**: `JsonKVStorage(BaseKVStorage)`
- **File**: `kv_store_{namespace}.json`
- **Methods**: `get_by_id()`, `upsert()`, `filter_keys()`

#### 2. NanoVectorDBStorage
- **Class**: `NanoVectorDBStorage(BaseVectorStorage)`
- **File**: `vdb_{namespace}.json`
- **Backend**: `NanoVectorDB` library
- **Methods**: `upsert()`, `query()`
- **Features**: Cosine similarity search, embedding-based

#### 3. HypergraphStorage
- **Class**: `HypergraphStorage(BaseHypergraphStorage)`
- **File**: `hypergraph_{namespace}.hgdb`
- **Backend**: `HypergraphDB` library
- **Methods**: 
  - Vertex: `upsert_vertex()`, `get_vertex()`, `has_vertex()`
  - Hyperedge: `upsert_hyperedge()`, `get_hyperedge()`, `has_hyperedge()`
  - Navigation: `get_nbr_v_of_vertex()`, `get_nbr_e_of_vertex()`, `vertex_degree()`, `hyperedge_degree()`

### LLM Integration (`hyperrag/llm.py`)

**Functions**:
- `openai_complete_if_cache()`: LLM completion with caching
- `openai_embedding()`: Text embeddings
- Retry logic with exponential backoff

### Utilities (`hyperrag/utils.py`)

**Key Functions**:
- `chunking_by_token_size()`: Text chunking
- `compute_mdhash_id()`: Hash-based IDs
- `truncate_list_by_token_size()`: Token-based truncation
- `limit_async_func_call()`: Rate limiting for async functions

---

## Query Modes Comparison

| Mode | Entity Retrieval | Relationship Retrieval | Graph Structure | Speed | Accuracy |
|------|-----------------|----------------------|-----------------|-------|----------|
| `naive` | ❌ | ❌ | ❌ | ⚡⚡⚡ | ⭐⭐ |
| `hyper-lite` | ✅ | ❌ | ✅ | ⚡⚡ | ⭐⭐⭐ |
| `hyper` | ✅ | ✅ | ✅ | ⚡ | ⭐⭐⭐⭐ |
| `graph` | ✅ | ✅ | ✅ | ⚡ | ⭐⭐⭐⭐ |
| `llm` | ❌ | ❌ | ❌ | ⚡⚡⚡ | ⭐ |

---

## Key Design Patterns

1. **Multi-level Retrieval**: Entity-level + Relationship-level for comprehensive context
2. **Hypergraph Structure**: Captures multi-entity relationships (not just pairs)
3. **Iterative Extraction**: Gleaning process to catch missed entities
4. **Summarization**: Automatic summarization when entity/relationship data exceeds token limits
5. **Vector + Graph**: Combines semantic search (vectors) with structured knowledge (graph)
6. **Caching**: LLM response caching to avoid redundant API calls

---

## Data Flow Summary

```
Document Insert:
  Text → Chunks → Entity Extraction (LLM) → Entities + Relationships
    → Hypergraph (vertices + hyperedges)
    → Vector DBs (entities, relationships, chunks)

Query:
  Query → Keyword Extraction (LLM) → Entity/Relationship Keywords
    → Vector Search → Hypergraph Navigation → Text Chunks
    → Context Building → LLM Response → Answer
```

---

## File Structure

```
hyperrag/
  ├── hyperrag.py          # Main HyperRAG class
  ├── operate.py            # Core operations (extraction, querying)
  ├── storage.py            # Storage implementations
  ├── llm.py                # LLM integration
  ├── prompt.py             # All prompts
  ├── base.py               # Base classes and schemas
  └── utils.py              # Utility functions

reproduce/
  ├── Step_0.py             # Dataset preprocessing
  ├── Step_1.py             # HyperGraph construction
  ├── Step_2_extract_question.py  # Question extraction
  └── Step_3_response_question.py # Answer generation
```

---

This architecture enables HyperRAG to:
1. **Extract structured knowledge** from unstructured text
2. **Build rich hypergraph** representations with multi-entity relationships
3. **Perform multi-level retrieval** (entity + relationship) for comprehensive context
4. **Generate accurate answers** using both semantic search and graph structure

