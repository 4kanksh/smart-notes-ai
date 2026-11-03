# Smart Notes AI — Project Explanation

## 1. Problem statement

Students often need to search long PDFs and class notes for specific answers. Smart Notes AI lets a user upload study material and ask questions in natural language. It can also create summaries and practice quizzes from the uploaded notes.

## 2. Concepts demonstrated

### Retrieval-Augmented Generation (RAG)

RAG combines document retrieval with language generation. The app first searches indexed document chunks, then provides the retrieved text to a local language model as context. The model is instructed to answer from that context and admit when the context is insufficient.

### Semantic search

Semantic search compares vector representations of the user's question and document chunks. This can find passages that are conceptually related even when the wording is different. It is not the same as a simple exact keyword search.

### Embeddings

The `sentence-transformers/all-MiniLM-L6-v2` model converts each text chunk and question into a numerical vector. Similar meanings tend to have vectors that are close in the embedding space.

### Vector database

ChromaDB stores the chunk text, embeddings, and metadata such as source filename and page number. The collection uses cosine distance for similarity comparisons.

### Chunking

PDF text is split into pieces of about 900 characters, with 150 characters of overlap. Chunking makes retrieval more focused, and overlap helps preserve context across chunk boundaries.

### Summarization

The summary tab supplies the indexed text to Ollama and asks for a short revision summary, a detailed chapter summary, or exam-focused key points.

### Quiz generation

The quiz tab asks the local language model to generate multiple-choice questions, options, answers, and brief explanations from the indexed notes. Ollama is given a JSON schema, so the output is structured data the app can show as clickable options. After the user submits, the app marks each answer and shows the final score.

### Context window

Ollama's default context window (about 2,048 tokens) is smaller than the summary and quiz prompts. When a prompt is too long, Ollama silently drops its beginning, which is where the instructions are. The app therefore requests an 8,192-token context window (`LLM_CONTEXT_TOKENS`).

## 3. Architecture

```text
PDF upload
   |
   v
pypdf extracts page text
   |
   v
Text chunking + source/page metadata
   |
   v
Sentence Transformers creates embeddings
   |
   v
ChromaDB stores vectors and chunks
   |
User question
   |
   v
Question embedding
   |
   v
ChromaDB semantic retrieval (top 4 chunks)
   |
   v
Retrieved passages + question sent to Ollama
   |
   v
Answer shown in Streamlit with source references
```

Summaries and quizzes use the indexed document text directly as context rather than retrieving only the top four chunks.

## 4. Main files

- `app.py`: Streamlit UI and the complete application pipeline.
- `requirements.txt`: Python dependencies.
- `RUN.md`: Setup and run instructions.
- `EXPLANATION.md`: Concepts, architecture, and workflow.
- `data/chroma_db/`: Local ChromaDB data created after indexing; it is not included in the project ZIP.

## 5. Important functions in `app.py`

- `split_text()`: Cleans and splits text into overlapping chunks.
- `read_pdfs()`: Extracts text from each PDF page and keeps page metadata.
- `index_documents()`: Embeds chunks and stores them in ChromaDB.
- `retrieve()`: Embeds a question and retrieves the nearest chunks.
- `answer_question()`: Builds a grounded prompt and asks Ollama to answer.
- `build_summary()`: Generates a summary from indexed text.
- `build_quiz()`: Generates MCQs from indexed text.

## 6. Why use a local stack?

- Python and the listed libraries are free to use.
- Streamlit provides a GUI without requiring separate JavaScript frontend code.
- ChromaDB runs locally and does not require a hosted vector database.
- Ollama runs the language model on the user's computer, avoiding a mandatory paid inference API.
- The embedding model is downloaded and run locally after the first setup.

The first-time setup downloads models and packages, so an internet connection is required initially. After downloading, core inference can run offline.

## 7. Limitations

- Scanned PDFs require OCR, which is not included in this first version.
- The quality of responses depends on PDF text quality, chunking, retrieval, and the selected LLM.
- Summarization and quiz generation are limited by the model's context capacity; this app uses a text limit when assembling document context.
- The app replaces the previous indexed collection when new PDFs are indexed.
- Local models can be slower on laptops with limited RAM.
- Source references identify retrieved document chunks and pages; they do not guarantee that every generated claim is correct.

## 8. Future improvements

- Add OCR support for scanned notes.
- Add hybrid keyword + vector search.
- Add conversation memory.
- Add downloadable summaries and quizzes.
- Add a feedback rating and retrieval evaluation.
- Add support for DOCX and TXT files.
- Add per-document selection and deletion.
