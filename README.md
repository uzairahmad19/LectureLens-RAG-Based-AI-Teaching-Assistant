# LectureLens - Video RAG System

A Retrieval-Augmented Generation (RAG) pipeline that turns lecture videos into a searchable, conversational knowledge base. Ask any question about your course and the system tells you exactly which video to watch and at what timestamp.


---

## What It Does

You type a question like *"How does groupby work in pandas?"* and the system:

1. Converts your question into a semantic vector using BGE-M3
2. Searches lecture transcript chunks using cosine similarity
3. Passes the most relevant chunks to LLaMA 3.2 as context
4. Returns a natural language answer with the exact video title and timestamp
5. Automatically evaluates the answer quality using **cosine similarity** — no external evaluation library or LLM judge required

---

## Project Structure

```
LectureLens/
│
├── videos/                        ← Place your course .mp4 files here
├── audios/                        ← Auto-created: extracted .mp3 files
<<<<<<< HEAD
├── transcripts/                   ← Auto-created: raw JSON transcript chunks
├── merged_transcripts/            ← Auto-created: merged JSON chunks
│
├── process_video.py               Stage 1 — Extract audio from videos
├── create_chunks.py               Stage 2 — Transcribe audio to text chunks
├── merge_chunks.py                Stage 2b — Merge small chunks into larger ones
├── read_chunks.py                 Stage 3 — Generate and save embeddings
├── process_query.py               Stage 4 — CLI query interface with evaluation
=======
├── transcripts/                   ← Auto-created: raw JSON chunks (one per Whisper segment)
├── merged_transcripts/            ← Auto-created: merged JSON chunks (5 segments → 1 chunk)
│
├── process_video.py               Stage 1 — Extract audio from videos
├── create_chunks.py               Stage 2 — Transcribe audio to text chunks
├── merge_chunks.py                Stage 3 — Merge small chunks into larger ones
├── read_chunks.py                 Stage 4 — Generate and save embeddings
├── process_query.py               Stage 5 — CLI query interface
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4
│
├── app.py                         Flask API (query endpoint)
├── index.html                     Frontend demo
│
├── chunks_with_embeddings.joblib  Auto-created: embedded chunk database
└── eval_log.json                  Auto-created: evaluation scores log (CLI)
```

---

## Pipeline Overview

![LectureLens](https://github.com/user-attachments/assets/f432d5c5-06ff-44de-bb4b-c778dab88a33)



| Stage | File | Tool | What it does |
|-------|------|------|--------------|
| 1 | `process_video.py` | FFmpeg | Extracts audio from `.mp4` lecture files |
| 2 | `create_chunks.py` | Faster-Whisper | Transcribes audio into timed text segments |
<<<<<<< HEAD
| 2b | `merge_chunks.py` | — | Merges small segments into larger chunks (groups of 5) |
| 3 | `read_chunks.py` | BGE-M3 via Ollama | Embeds each chunk into a 1024-dim vector |
| 4 | `app.py` | LLaMA 3.2 via Ollama | Retrieves chunks, generates answers, evaluates quality |
=======
| 3 | `merge_chunks.py` | Python | Merges every 5 segments into one larger chunk |
| 4 | `read_chunks.py` | BGE-M3 via Ollama | Embeds each merged chunk into a 1024-dim vector |
| 5 | `app.py` | LLaMA 3.2 via Ollama | Retrieves chunks and generates answers |
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

---

## Getting Started

### 1. Prerequisites

Install the following before running anything:

- **Python 3.10+**
- **FFmpeg** — [ffmpeg.org](https://ffmpeg.org/download.html) — must be on your system PATH
- **Ollama** — [ollama.com](https://ollama.com) — runs LLMs locally

Pull the required models through Ollama:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

### 2. Install Python dependencies

```bash
pip install faster-whisper flask flask-cors numpy pandas joblib scikit-learn requests
```

No additional evaluation libraries are needed — the evaluation is built on top of BGE-M3 and scikit-learn, which are already in the stack.

### 3. Add your course videos

Download your lecture videos and place all `.mp4` files inside the `videos/` folder:

```
videos/
  Part 1 - Introduction to Pandas.mp4
  Part 2 - DataFrames and Series.mp4
  Part 3 - Data Cleaning.mp4
  ...
```

> **Naming tip:** The pipeline automatically extracts the episode number from filenames that contain `Part`, `Episode`, `Lesson`, or `Ep` followed by a number — e.g. `Part 3`, `Episode 7`, `Lesson 12`. Make sure your filenames follow this pattern.

---

## Running the Pipeline

Run these scripts **once**, in order. After that you only need `app.py`.

### Stage 1 — Extract audio

```bash
python process_video.py
```

Reads every `.mp4` from `videos/`, extracts the audio track using FFmpeg, and saves `.mp3` files into `audios/`.

### Stage 2 — Transcribe

```bash
python create_chunks.py
```

Loads each `.mp3` from `audios/`, transcribes it using Faster-Whisper, and saves timed text chunks as JSON files in `transcripts/`. Each JSON file contains one entry per Whisper segment — these are short, typically 3–8 seconds each.

> ⚠️ This step takes time depending on how many videos you have. On low-end hardware, expect roughly 1–3× real-time (a 30-minute lecture may take 30–90 minutes to transcribe).

<<<<<<< HEAD
### Stage 2b — Merge chunks
=======
### Stage 3 — Merge chunks
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

```bash
python merge_chunks.py
```

<<<<<<< HEAD
Reads the raw JSON files from `transcripts/` and merges every 5 consecutive segments into a single larger chunk, saving the results into `merged_transcripts/`. This reduces the total number of chunks and gives the embedding model more context per unit.

### Stage 3 — Generate embeddings
=======
Reads the raw Whisper segments from `transcripts/` and merges every 5 consecutive segments into a single larger chunk, saving the results into `merged_transcripts/`. The merged chunk spans the `start` time of the first segment to the `end` time of the fifth, and combines all their text into one block.

This step exists for two reasons. First, individual Whisper segments are very short and often mid-sentence — they don't contain enough context for the embedding model to understand the topic. Second, fewer but larger chunks means fewer embedding vectors to store and compare, which is important on low-end hardware.

> You can adjust how many segments get merged by changing `n = 5` at the top of `merge_chunks.py`. Higher values produce broader chunks with more context but less precise timestamps; lower values give tighter timestamps but may not capture enough meaning.

### Stage 4 — Generate embeddings
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

```bash
python read_chunks.py
```

<<<<<<< HEAD
Reads all JSON files from `merged_transcripts/`, sends each chunk to BGE-M3 via Ollama to generate a 1024-dimensional embedding vector, and saves everything into `chunks_with_embeddings.joblib`.
=======
Reads all merged JSON chunks from `merged_transcripts/`, sends each chunk's text to the BGE-M3 model via Ollama to generate a 1024-dimensional embedding vector, and saves everything into `chunks_with_embeddings.joblib`.
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

> ⚠️ This also takes time on first run. The `.joblib` file is reused on every query — you never need to run this again unless you add new videos.

### Stage 5 — Start the API

```bash
python app.py
```

Starts the Flask server on `http://localhost:5000`. Keep this running while using the frontend.

---

## Running the Frontend

Open a second terminal in the project folder and run:

```bash
python -m http.server 8080
```

Then open your browser and go to:

```
http://localhost:8080
```

> **VS Code users:** Install the **Live Server** extension by Ritwick Dey, then right-click `index.html` → *Open with Live Server*. It auto-reloads when you save the file.

---

## CLI Usage

You can also query the system directly from the terminal without the frontend:

```bash
python process_query.py
```

After the answer is printed, evaluation scores are shown immediately and appended to `eval_log.json` for tracking quality over time.

---

## API Reference

### `POST /query`

Takes a natural language question, returns the LLM-generated answer, and automatically evaluates its quality using cosine similarity.

**Request:**
```json
{ "question": "How does groupby work in pandas?" }
```

**Response:**
```json
{
  "answer": "This is covered in Video 5 — GroupBy and Aggregation, at 3 minutes 22 seconds...",
  "evaluation": {
    "answer_relevancy": 0.87,
    "faithfulness": 0.91
  }
}
```

The `evaluation` field is always present. If embedding fails for any reason, it will contain `{"error": "..."}` instead of scores, but the `answer` is always returned regardless.

---

## Evaluation

### How it works

After every query, two quality scores are computed using the same BGE-M3 embeddings already used for retrieval. No extra libraries, no LLM judge, no ground truth labels needed.

| Metric | Comparison | What it tells you |
|--------|------------|-------------------|
| **Answer Relevancy** | question ↔ answer | Is the answer semantically on-topic for the question? Low = vague or off-topic response |
| **Faithfulness** | answer ↔ retrieved chunks (averaged) | Is the answer grounded in the source material? Low = answer diverges from the chunks (hallucination risk) |

Both scores are in the range **0.0 – 1.0**.

### How scores are computed

```
answer_relevancy = cosine_similarity( embed(question), embed(answer) )

faithfulness     = mean( cosine_similarity( embed(answer), embed(chunk_i) )
                         for each retrieved chunk )
```

### Interpreting scores

| Score | Meaning |
|-------|---------|
| ≥ 0.70 | 🟢 Good |
| ≥ 0.50 | 🟡 Acceptable |
| < 0.50 | 🔴 Poor |

### Score log (CLI)

When running `process_query.py`, scores for every query are appended to `eval_log.json`:

```json
[
  {
    "query": "How does groupby work in pandas?",
    "scores": {
      "answer_relevancy": 0.87,
      "faithfulness": 0.91
    }
  }
]
```

<<<<<<< HEAD
This file accumulates across sessions, making it easy to compare quality over multiple queries for your project report.
=======
**Response:**
```json
{
  "num_queries": 1,
  "top_k": 5,
  "mrr": 1.0,
  "precision_at_k": 0.8,
  "per_query": [...]
}
```

A chunk is counted as **relevant** if its text contains any of the `relevant_keywords` (case-insensitive). The frontend automatically extracts keywords from your query and calls this endpoint after every question.

---

## Screenshots

**Landing Page:**

<img width="1920" height="910" alt="homepage" src="https://github.com/user-attachments/assets/5a89fc37-01e8-474f-9a2b-0ae4df21499e" />

**ChatBox:**

<img width="1920" height="910" alt="demo" src="https://github.com/user-attachments/assets/b494c1f4-10bf-4f95-ba70-117e7c6bc427" />

**Sample Query & Response:**

<img width="1920" height="909" alt="response loading" src="https://github.com/user-attachments/assets/e29a25f4-5b01-4666-bf8c-1ec4b4ada1a5" />
<img width="1920" height="800" alt="output" src="https://github.com/user-attachments/assets/9e5e58b7-9fa6-400d-9c53-8f4544094606" />

**Evaluation:**

<img width="1920" height="817" alt="evaluation" src="https://github.com/user-attachments/assets/80d4a98b-ed3e-44b9-8419-84429e6c1dc4" />



## Evaluation Metrics

| Metric | Formula | What it means |
|--------|---------|---------------|
| **MRR** (Mean Reciprocal Rank) | `1 / rank_of_first_relevant_chunk` | How highly the first relevant chunk is ranked. 1.0 = always #1 |
| **Precision@5** | `relevant_chunks_in_top_5 / 5` | Fraction of the 5 retrieved chunks that were actually relevant |

**Interpreting scores:**

| Score | MRR | Precision@5 |
|-------|-----|-------------|
| 🟢 Good | ≥ 1.0 | ≥ 0.6 |
| 🟡 Acceptable | ≥ 0.5 | ≥ 0.4 |
| 🔴 Poor | < 0.5 | < 0.4 |
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

---

## Low-End Hardware Optimisations

This project was developed and tested on a **low-spec machine**. Several deliberate choices were made to keep it runnable without a GPU or large amounts of RAM:

| Optimisation | Detail |
|---|---|
| **Faster-Whisper instead of original Whisper** | Uses CTranslate2 under the hood — roughly 4× faster on CPU with identical accuracy |
| **int8 quantisation** | `compute_type="int8"` in `create_chunks.py` halves Whisper's memory usage with minimal quality loss |
| **`base` Whisper model** | Chosen over `small`, `medium`, or `large` — accurate enough for clear lecture audio, much faster to run |
| **CPU device mode** | `device="cpu"` in Faster-Whisper — no GPU required |
| **LLaMA 3.2 (3B)** | Smallest capable LLaMA model — runs on 4–6 GB RAM via Ollama without GPU offloading |
| **BGE-M3 via Ollama** | Embedding inference handled locally by Ollama, avoiding any cloud dependency or GPU requirement |
| **Chunk merging (5 segments → 1)** | `merge_chunks.py` groups 5 short Whisper segments into one chunk before embedding — reduces total vector count significantly, lowering both RAM usage and cosine similarity computation time per query |
| **Embeddings pre-computed and cached** | All chunk embeddings are computed once and saved to `.joblib`. Every query loads from disk instantly — no re-embedding on each request |
| **Top-K = 5** | Only the 5 most relevant chunks are passed to the LLM, keeping prompt size and generation time low |
| **Non-streaming generation** | `"stream": False` in Ollama calls — simpler to handle and avoids incremental token overhead on slow machines |
| **Evaluation reuses existing embeddings** | Quality scores are computed using the same BGE-M3 model already running — no second model or extra memory needed |

---

## Technologies Used

| Component | Technology |
|-----------|------------|
| Audio extraction | FFmpeg |
| Speech-to-text | Faster-Whisper (`base` model, int8, CPU) |
| Embedding model | BGE-M3 (via Ollama) |
| Language model | LLaMA 3.2 (via Ollama) |
| Similarity search | Cosine similarity (scikit-learn) |
| Answer evaluation | Cosine similarity on BGE-M3 embeddings (no extra libraries) |
| Backend API | Flask + Flask-CORS |
| Data storage | pandas DataFrame + joblib |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## Troubleshooting

**FFmpeg not found**
Make sure FFmpeg is installed and on your system PATH. Test with `ffmpeg -version` in a terminal.

**Ollama connection refused**
Make sure Ollama is running before starting `app.py`. Run `ollama serve` in a separate terminal, or check that the Ollama desktop app is open.

**`chunks_with_embeddings.joblib` not found**
<<<<<<< HEAD
You need to run all pipeline scripts (`process_video.py` → `create_chunks.py` → `merge_chunks.py` → `read_chunks.py`) before starting the API.
=======
You need to run all four pipeline scripts in order (`process_video.py` → `create_chunks.py` → `merge_chunks.py` → `read_chunks.py`) before starting the API.
>>>>>>> 7cfc170c3fe6bfb7b9e0c91c8e7b501d4f0309a4

**Transcription is very slow**
This is expected on CPU. Consider running overnight for large video collections. Each file is skipped automatically if already transcribed, so you can stop and resume safely.

**LLM response is slow**
LLaMA 3.2 on CPU can take 30–90 seconds per response depending on your hardware. This is normal for local inference without a GPU.

**Evaluation scores seem low**
A faithfulness score below 0.5 usually means the answer contains information not present in the retrieved chunks. A low answer relevancy score means the response did not address the question directly. Both are useful signals for debugging retrieval or prompt quality.

---

## Folder Setup Checklist

Before running the pipeline, make sure this is in place:

- [ ] `videos/` folder exists and contains your `.mp4` lecture files
- [ ] FFmpeg is installed and accessible from terminal
- [ ] Ollama is running with `bge-m3` and `llama3.2` pulled
- [ ] Python dependencies are installed
- [ ] `audios/`, `transcripts/`, and `merged_transcripts/` folders will be created automatically

---
## Future Scope

### 1. GPU Acceleration
The entire pipeline currently runs on CPU. Adding GPU support for Faster-Whisper (`device="cuda"`) and switching to a larger Whisper model (`medium` or `large-v2`) would significantly improve transcription accuracy and speed. Similarly, running BGE-M3 and LLaMA 3.2 with GPU offloading via Ollama would reduce query response time from minutes to seconds.

### 2. Larger and Quantised Language Models
LLaMA 3.2 (3B) was chosen for low-end hardware compatibility. As hardware improves, upgrading to LLaMA 3.1 (8B) or Mistral 7B would produce more detailed and accurate answers.

### 3. Hybrid Search (Keyword + Semantic)
The current retrieval is purely semantic (dense vector search). Adding BM25 keyword search and combining it with cosine similarity through a re-ranker (such as a cross-encoder) would improve results for queries with specific technical terms, variable names, or function names that semantic search alone may miss.

### 4. Multi-Course Support
Currently the system is scoped to a single course. A multi-course architecture would allow students to upload and switch between different subjects. This would require a per-course vector store, a course selection UI, and a routing layer in the Flask API that queries only the relevant embeddings.

### 5. Automatic Slide and PDF Integration
Lecture videos are often accompanied by slides or PDF notes. Integrating PDF parsing (via PyMuPDF or pdfplumber) and embedding slide content alongside video transcripts would give the system a richer knowledge base — allowing it to answer questions that appear in slides but are not spoken aloud in the video.

### 6. Speaker Diarisation
Faster-Whisper does not distinguish between speakers. Adding speaker diarisation (via pyannote.audio) would allow the system to identify when a student asks a question versus when the instructor answers — improving the quality of the transcript and allowing filtered search by speaker role.

### 7. Web-Based Upload Interface
Currently, videos must be manually placed in the `videos/` folder and the pipeline scripts must be run from the terminal. A drag-and-drop web interface for uploading videos and triggering the pipeline automatically would make the system accessible to non-technical users such as educators.

### 8. Summarisation per Video
Each video could be automatically summarised into a set of key topics after transcription. These summaries could be displayed in the frontend as a table of contents, letting users browse what a video covers before asking questions — reducing dependency on exact query wording.

### 9. User Feedback Loop
Adding a thumbs up / thumbs down button on each answer would allow users to flag poor responses. Over time, this feedback could be used to fine-tune the prompt, adjust chunk size, or retrain a re-ranker — turning the system into a self-improving retrieval engine.

### 10. Cloud Deployment
The system is fully local by design. A natural next step would be containerising it with Docker and deploying it on a cloud platform (AWS EC2, Google Cloud Run, or Hugging Face Spaces) with the `.joblib` file stored in an object store like AWS S3 — making it accessible to an entire class rather than just one machine.

---
*LectureLens - RAG based AI teaching Agent*
