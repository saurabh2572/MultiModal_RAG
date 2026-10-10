# MultiModal RAG

PDF ingestion creates three retrieval ready outputs: positioned elements, modality aware chunks, and rendered figure crops. Place the Tata Motors annual report at `data/raw/tata_motors_doc.pdf` before running the commands below. The source PDF is kept locally and is not included in the repository.

```powershell
uv sync
uv run python main.py --max-pages 5
uv run python main.py
```

The first command installs the locked environment. The second processes five pages as a trial; omit `--max-pages` to process the whole PDF. For a specific range, use `--start-page 350 --max-pages 1`. Outputs appear in `data/processed/tata_motors_doc/`:

- `elements.jsonl`: text blocks, structured table rows, and image OCR records with page bounding boxes.
- `chunks.jsonl`: combined text blocks, row preserving table chunks, and image chunks with OCR and captions. Each record has a stable ID and source metadata.
- `images/`: visible figure crops, named by PDF page and image index.
- `manifest.json`: source and extraction counts.

`--ocr auto` runs OCR on images and pages with little native text. `--ocr all` also OCRs every page. `--ocr off` disables OCR. Images are rendered at 150 DPI by default; increase with `--dpi` for small print. Table and text chunk sizes are controlled by `--max-tokens` and `--overlap-tokens` (token counts are estimates).

The extraction uses PyMuPDF for text, table boundaries, and image position, with RapidOCR on ONNX Runtime for OCR. This is a local parser; no `.env` keys are needed for ingestion. The older `Async_Modern_History_RAG_Chatbot` project supplied the staged extraction and metadata pattern, while this project adds separate table and image records.

## Modality-based vector database

After parsing, build four independent FAISS collections from `chunks.jsonl`:

| Collection | Input | Embedding model |
| --- | --- | --- |
| `text` | Narrative chunks | BAAI/bge-small-en-v1.5 |
| `table` | Table chunks with units and nearby headings | BAAI/bge-small-en-v1.5 |
| `image_text` | Figure captions and OCR text when available | BAAI/bge-small-en-v1.5 |
| `image_visual` | Cropped image pixels | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`) |

These vector spaces stay separate because BGE and CLIP embeddings are not directly comparable. All vectors are normalized and stored in FAISS inner-product indexes. Each collection includes `records.jsonl`, which maps vector rows to chunk IDs and citation metadata. The database root also has `manifest.json` with model names, dimensions, counts, and the source chunk checksum.

Run a small trial in its own directory, then build the complete database:

```powershell
uv run python build_vector_db.py --max-chunks 30 --output-dir data/vector_db/trial
uv run python build_vector_db.py
```

The BGE model is already cached on this machine. If it is not available later, add `--allow-model-download`. OpenCLIP downloads its pretrained weights on first use unless you pass a local checkpoint path through `--clip-pretrained`. Existing output directories are never overwritten; choose a new `--output-dir` to rebuild.

Query one collection at a time:

```powershell
uv run python search_vector_db.py --space table --query "deferred tax liabilities"
uv run python search_vector_db.py --space image_visual --query "electric vehicle"
uv run python search_vector_db.py --space image_visual --image-query path/to/query.png
```

The build and query files are provided for you to run; creating them does not start embedding or download model weights.

## Online query workflow

The Streamlit app uses LangGraph to run authentication, query understanding, parallel dense and SQLite FTS5/BM25 retrieval, reciprocal-rank fusion, cross-encoder reranking, neighboring-chunk context expansion, and a grounded answer with numbered citations. Google Gemini can receive up to two retrieved figure crops with the textual evidence; the Ollama alternative currently answers from text/OCR evidence only. LangSmith tracing is controlled by `LANGSMITH_TRACING`, and the existing `LANGCHAIN_*` aliases in `.env` are supported.

First build the *full* vector database and matching keyword index. The online workflow rejects a trial vector database made with `--max-chunks`; a trial shares the file checksum but omits many chunks.

```powershell
uv sync
uv run python build_vector_db.py
uv run python build_search_index.py
```

To run ingestion and both index builds as one workflow, use `uv run python build_pipeline.py`. It parses the PDF first, then starts the independent vector and BM25 builds concurrently with `asyncio`. Choose unused `--vector-db-dir` and `--keyword-db-path` values when rebuilding; index outputs are never overwritten.

Copy the new `RAG_*` settings from `.env.example` into your existing `.env` without replacing its API keys. Set a strong `RAG_ACCESS_TOKEN`; the app fails closed when it is absent. Then start the frontend:

```powershell
uv run streamlit run streamlit_app.py
```

The access token is a local demo gate, not a multi-user identity/authorization system. Keep Streamlit bound to localhost or put it behind production authentication before exposing it publicly. The token is checked before invoking LangGraph and is not included in graph state or LangSmith traces. Enabling tracing may transmit user questions, retrieved document excerpts, and answers to LangSmith; leave it off if the document is confidential and that transfer is not approved.

The generated FAISS and BM25 indexes are local files and are not built by starting Streamlit. The app loads those indexes at startup; embedding/reranking inference and LLM calls happen when you run a query. The online LangGraph path uses async nodes: dense retrieval, BM25 search, cross-encoder reranking, and figure encoding run in a bounded four-thread pool, while the LLM uses native `ainvoke()`. Dense and BM25 branches run concurrently. Local model access and the shared SQLite connection are synchronized for overlapping requests. To check the orchestration without downloads or API calls, run:

```powershell
uv run python -m unittest tests.test_online_workflow
```
