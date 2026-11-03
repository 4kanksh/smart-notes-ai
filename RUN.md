# Smart Notes AI — Run Guide

## What this app does

- Uploads one or more text-based PDFs.
- Splits PDF text into chunks and creates embeddings.
- Stores vectors locally in ChromaDB.
- Retrieves relevant chunks using semantic similarity.
- Uses Ollama to answer questions based on retrieved passages.
- Generates summaries and multiple-choice quizzes from indexed notes.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- VS Code with the Python extension
- Ollama installed
- Enough free RAM and disk space for a local language model and the embedding model

## 1. Install Ollama and download a model

Install Ollama from: https://ollama.com/download

Open a new terminal and run:

```powershell
ollama --version
ollama pull llama3.2:1b
```

Test the model:

```powershell
ollama run llama3.2:1b
```

Ask a simple question, then type `/bye` to exit. Keep Ollama running when using the app. If the app cannot connect, reopen Ollama from the Start menu and retry.

## 2. Open the project

Extract the project ZIP, then open the `smart_notes_ai` folder in VS Code.

Open **Terminal → New Terminal**. Make sure the terminal's current directory is the folder containing `app.py`.

## 3. Create a virtual environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Or run the virtual-environment Python directly without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Start the application

```bash
python -m streamlit run app.py
```

Streamlit normally opens the app in your browser at `http://localhost:8501`.

## 5. Use the app

1. In the left sidebar, keep the model name as `llama3.2:1b`, or enter a model you have downloaded.
2. Upload one or more text-based PDF files.
3. Click **Index PDFs** and wait for the indexing confirmation.
4. In **Ask your notes**, enter a question and click **Get answer**.
5. Open **View retrieved passages** to inspect the source passages and page numbers.
6. Use **Summarize** to generate a revision summary.
7. Use **Generate quiz** to create practice MCQs, select your answers, and click **Submit answers** to see your score.

## Troubleshooting

### `ollama` is not recognized

Install Ollama, close and reopen the terminal, and run `ollama --version` again. On Windows, Ollama should be installed and available in your PATH.

### The app cannot connect to Ollama

Open the Ollama desktop app or start the Ollama service, then retry. Test the model with `ollama run llama3.2:1b`.

### Model not found

Run `ollama pull llama3.2:1b`, or change the model name in the app to one you have installed.

### Installation errors for Python packages

Use Python 3.10 or 3.11 in a fresh virtual environment. If pip reports a package compatibility error, check the Python version with `python --version`.

### No text found in the PDF

The current version extracts selectable text. Scanned/image-only PDFs need OCR before they can be indexed. Try a normal text-based PDF first.

### First run is slow

The embedding model downloads the first time it runs. Ollama also downloads its language model separately. Later runs reuse the downloaded models.

### Laptop is slow or runs out of memory

Close other heavy apps and use a smaller Ollama model supported by your machine. Model speed depends on RAM, CPU/GPU, and model size.

## Important notes

- This is a local demonstration, not a production multi-user service.
- Indexing a new upload replaces the current indexed collection.
- The app stores ChromaDB files under `data/chroma_db/`.
- The model may occasionally produce incorrect answers. Check the retrieved passages and source documents.
- `pypdf` does not perform OCR.
