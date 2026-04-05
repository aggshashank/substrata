# Substrata

Substrata is a local-first AI research wiki powered by Ollama. It turns source documents into clean text, splits them into retrieval-friendly chunks, and stores embeddings in a local ChromaDB vector store for later RAG workflows.

The project is inspired by Andrej Karpathy's recent post on LLM knowledge bases and adapts that local knowledge-layer idea into an offline-first personal research workflow.

The repository is currently implemented through Phase 3 of the execution plan. That means the foundation, parsing, chunking, markdown generation, and vector-store layers are in place. The ingestion pipeline, chat interface, and CLI are planned but not added yet.

## Current Capabilities

- Load project settings from `.env` via `config.py`
- Track processed files, embedding runs, and run metadata in SQLite
- Parse `.pdf`, `.html`, `.htm`, `.md`, and `.txt` files into normalized text
- Split text into overlapping sentence-aware chunks
- Generate Obsidian-compatible markdown entries for papers and concepts
- Create and query embeddings locally through Ollama + LiteLLM + ChromaDB

## Project Structure

```text
substrata/
|-- config.py
|-- requirements.txt
|-- sources/
|-- wiki/
|-- data/
`-- utils/
    |-- chunking.py
    |-- embeddings.py
    |-- llm.py
    |-- markdown.py
    |-- parsers.py
    `-- storage.py
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally
- The Ollama models referenced in `.env`
  - `ollama/llama3.2:3b`
  - `ollama/nomic-embed-text`

## Setup

```bash
copy .env.example .env
pip install -r requirements.txt
```

Then make sure Ollama is running and the configured models are available locally.

## Configuration

Default environment values:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=ollama/llama3.2:3b
EMBED_MODEL=ollama/nomic-embed-text
WIKI_DIR=./wiki
SOURCES_DIR=./sources
CHROMA_DIR=./data/chroma
DB_PATH=./data/registry.db
MAX_LLM_CALLS_PER_RUN=50
CHUNK_SIZE=750
CHUNK_OVERLAP=100
LOG_LEVEL=INFO
```

## Quick Verification

Initialize the project directories and vector store:

```bash
python -c "from config import get_settings; get_settings().ensure_dirs(); from utils.embeddings import WikiVectorStore; print(WikiVectorStore().get_stats())"
```

Try the parser and chunker on a local text file:

```bash
python -c "from pathlib import Path; from utils.parsers import parse_file; from utils.chunking import chunk_text; text = parse_file(Path('sources/example.txt')); print(len(chunk_text(text, doc_id='example', source_file='sources/example.txt')))"
```

If Ollama is offline, embedding calls fail safely and return empty results instead of crashing.

## Module Notes

`utils/storage.py`
Stores processed file records, embedding logs, and run logs in SQLite.

`utils/parsers.py`
Extracts normalized text from supported document formats.

`utils/chunking.py`
Builds overlapping sentence-aware chunks using token estimates from word counts.

`utils/markdown.py`
Generates wiki markdown pages and writes them atomically.

`utils/llm.py`
Wraps LiteLLM completion calls and provides a basic Ollama health check.

`utils/embeddings.py`
Creates embeddings through Ollama and persists/query chunks in ChromaDB.

## Roadmap

- Phase 4: ingestion pipeline and prompt files
- Phase 5: maintenance workflows
- Phase 6: chat interface
- Phase 7: CLI and release polish
