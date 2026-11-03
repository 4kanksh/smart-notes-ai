# 📚 Smart Notes AI

A local-first study assistant that answers questions about your PDF notes, summarizes them, and quizzes you on them. It uses **Retrieval-Augmented Generation (RAG)**: answers are grounded in the uploaded documents and show the source file and page.

Everything runs on your own computer. There is no paid API, and your notes never leave the machine.

## Features

- **Ask your notes**: ask questions in plain language and get answers based only on the uploaded PDFs, with source pages and the retrieved passages shown for checking.
- **Semantic search**: finds relevant passages by meaning, not just matching keywords.
- **Summaries**: short revision summary, detailed chapter summary, or key points for an exam.
- **Interactive quiz**: generates multiple-choice questions you answer by clicking, then shows your score, the correct answers, and a short explanation for each.
- **Fully local**: the language model runs through [Ollama](https://ollama.com), and vectors are stored in a local ChromaDB database.

## How it works

```text
INDEXING
PDF ──► pypdf extracts text per page
    ──► split into ~900-character chunks (150-character overlap), tagged with file + page
    ──► Sentence Transformers (all-MiniLM-L6-v2) turns each chunk into a 384-number vector
    ──► ChromaDB stores text + vector + metadata on disk

ASKING A QUESTION
Question ──► embedded with the same model
         ──► ChromaDB returns the 4 closest chunks (cosine distance)
         ──► chunks + question are sent to the LLM (Llama 3.2 via Ollama)
         ──► answer is shown with its sources
```

The summary and quiz tabs send the indexed text (up to 18,000 characters) straight to the LLM. The quiz uses Ollama's structured JSON output, so each question, its four options and the correct answer can be shown as clickable choices and scored.

## Tech stack

| Component | Tool | Role |
|---|---|---|
| User interface | [Streamlit](https://streamlit.io) | Web app written in pure Python |
| PDF reading | [pypdf](https://pypi.org/project/pypdf/) | Extracts text from each page |
| Embeddings | [Sentence Transformers](https://www.sbert.net) `all-MiniLM-L6-v2` | Converts text into meaning vectors |
| Vector database | [ChromaDB](https://www.trychroma.com) | Stores vectors and runs similarity search |
| Language model | [Ollama](https://ollama.com) + `llama3.2:1b` | Generates answers, summaries and quizzes |

## Getting started

### Requirements

- Python 3.10 or 3.11
- [Ollama](https://ollama.com/download)
- About 4 GB of free disk space for the Python packages and models

### 1. Download the language model

```bash
ollama pull llama3.2:1b
```

### 2. Install the Python packages

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>. On first use it downloads the embedding model (about 90 MB).

## Usage

1. In the sidebar, upload one or more **text-based** PDFs and click **Index PDFs**.
2. **Ask your notes**: type a question and click **Get answer**. Open **View retrieved passages** to see the source chunks, their pages, and their similarity scores.
3. **Summarize**: pick a style and click **Generate summary**.
4. **Generate quiz**: choose the number of questions and difficulty, click **Generate quiz**, select your answers, then click **Submit answers** to see your score.

To get better answers and quizzes, install a larger model (for example `ollama pull llama3.2:3b`) and type its name into the **Ollama model** box in the sidebar.

## Project structure

```text
smart_notes_ai/
├── app.py            # Streamlit UI and the full RAG pipeline
├── requirements.txt  # Python dependencies
├── RUN.md            # Detailed setup and troubleshooting
├── EXPLANATION.md    # Concepts and architecture explained
└── data/chroma_db/   # Local vector database (created at runtime, not committed)
```

## Limitations

- Scanned or image-only PDFs need OCR, which is not included. Text inside images and diagrams is ignored.
- Indexing new PDFs replaces the previously indexed ones.
- Summaries and quizzes only use the first 18,000 characters of the indexed text.
- The default 1B-parameter model is fast but small. It can produce weak answers or mark the wrong quiz option as correct, so check its answers against the retrieved passages.
- This is a single-user local demo, not a production service.

## Future improvements

- OCR support for scanned notes
- Hybrid keyword + vector search
- A relevance threshold so unrelated questions get "not found" without calling the LLM
- Conversation memory for follow-up questions
- Adding documents without replacing the existing index
- DOCX and TXT support
- Downloadable summaries and quizzes

## More documentation

- [RUN.md](RUN.md): step-by-step setup and troubleshooting
- [EXPLANATION.md](EXPLANATION.md): RAG, embeddings, chunking and the architecture explained
