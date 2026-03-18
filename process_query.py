"""
Processes user query, retrieves relevant chunks, generates RAG answer using Ollama.
After each query, evaluates answer quality using cosine similarity on BGE-M3 embeddings.
No extra libraries needed — reuses the same embedding model already used for retrieval.

Usage: python process_query.py
Output: Prints answer and saves to response.txt. Appends scores to eval_log.json.
"""

import logging
import json
import os
import numpy as np
import joblib
import requests

from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE      = "chunks_with_embeddings.joblib"
RESPONSE_OUTPUT_FILE = "response.txt"
EVAL_LOG_FILE        = "eval_log.json"

# Ollama endpoints
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Models
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "llama3.2"

# Retrieval settings
TOP_K_RESULTS = 5
MAX_CHUNK_TEXT_CHARS = 400

COURSE_SUBJECT = "Data Analysis using Python"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


def create_embedding(text_list: list) -> list:
    """Embeds text list using Ollama."""
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": text_list},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


# Alias used by the evaluator for clarity
embed_many = create_embedding


def run_inference(prompt: str) -> str:
    """Sends prompt to LLaMA model via Ollama and returns response."""
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def retrieve_top_chunks(df, query_embedding: list, top_k: int):
    """Retrieves top-K chunks by cosine similarity to query embedding."""
    chunk_matrix = np.vstack(df["embedding"].values)
    query_vector = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(chunk_matrix, query_vector).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    result_df = df.loc[top_indices].copy()
    result_df["start"] = result_df["start"].apply(format_timestamp)
    result_df["end"] = result_df["end"].apply(format_timestamp)
    result_df["text"] = result_df["text"].str[:MAX_CHUNK_TEXT_CHARS]

    return result_df


def build_prompt(query: str, context_df) -> str:
    """Builds RAG prompt with retrieved context and query."""
    context_json = context_df[["name", "number", "start", "end", "text"]].to_json(orient="records")

    prompt = f"""You are a helpful teaching assistant for a course on {COURSE_SUBJECT}.

Below are transcript segments from the course videos. Each entry includes:
- "name": the video title
- "number": the video episode number
- "start" / "end": the timestamp range (MM:SS format)
- "text": what is being discussed at that time

Transcript segments:
{context_json}

---------------------------------
Student's question: "{query}"

Instructions:
- Answer naturally and conversationally — do NOT mention the JSON format.
- Tell the student which video(s) cover their question and at what timestamp.
- Reference the video by its title and episode number for clarity.
- If the question is unrelated to the course content, politely say you can only
  answer questions about {COURSE_SUBJECT}.
- Use "minutes and seconds" format when citing timestamps (e.g. "at 5 minutes 12 seconds").
"""
    return prompt


# ---------------------------------------------------------------------------
# Cosine similarity evaluation (no LLM, no extra dependencies)
# ---------------------------------------------------------------------------

def evaluate_with_cosine(query: str, answer: str, chunk_texts: list) -> dict:
    """
    Evaluates answer quality using cosine similarity on BGE-M3 embeddings.
    Reuses create_embedding() already used for retrieval — no new libraries needed.

    Two metrics:

    answer_relevancy  (question ↔ answer)
    ──────────────────────────────────────
    How semantically close is the generated answer to the original question?
    High = answer is on-topic. Low = answer is vague or off-topic.
    Proxy for: RAGAS Answer Relevancy.

    faithfulness  (answer ↔ retrieved chunks, averaged)
    ────────────────────────────────────────────────────
    How closely does the answer language match the retrieved source chunks?
    High = answer is grounded in the retrieved material.
    Low  = answer diverges from the chunks (hallucination risk).
    Proxy for: RAGAS Faithfulness.

    Both scores are in the range 0.0 – 1.0.
    """
    try:
        logger.info("Running cosine similarity evaluation...")

        # Embed question and answer individually
        question_vec = np.array(create_embedding([query])[0])
        answer_vec   = np.array(create_embedding([answer])[0])

        # answer_relevancy: question ↔ answer
        answer_relevancy = float(
            cosine_similarity([question_vec], [answer_vec])[0][0]
        )

        # faithfulness: answer ↔ each chunk, then averaged
        chunk_vecs   = np.array(embed_many(chunk_texts))
        chunk_sims   = cosine_similarity([answer_vec], chunk_vecs)[0]
        faithfulness = float(np.mean(chunk_sims))

        scores = {
            "answer_relevancy": round(answer_relevancy, 4),
            "faithfulness":     round(faithfulness,     4),
        }
        return scores

    except Exception as e:
        logger.error("Cosine evaluation failed: %s", e)
        return {"error": str(e)}


def save_eval_log(query: str, scores: dict) -> None:
    """Appends evaluation scores for this query to eval_log.json."""
    log = []
    if os.path.exists(EVAL_LOG_FILE):
        with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []

    log.append({"query": query, "scores": scores})

    with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation scores saved to '%s'.", EVAL_LOG_FILE)


def print_eval_scores(scores: dict) -> None:
    """Prints cosine evaluation scores with interpretation."""
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY EVALUATION")
    print("=" * 60)

    if "error" in scores:
        print(f"  Evaluation failed: {scores['error']}")
        print("=" * 60 + "\n")
        return

    def rating(s):
        if s >= 0.7:   return "Good"
        elif s >= 0.5: return "Acceptable"
        else:          return "Poor"

    ar = scores.get("answer_relevancy", 0)
    fi = scores.get("faithfulness",     0)

    print(f"  Answer Relevancy : {ar:.4f}  [{rating(ar)}]")
    print(f"    → Question ↔ Answer similarity")
    print(f"    → High = answer is on-topic for the question")
    print()
    print(f"  Faithfulness     : {fi:.4f}  [{rating(fi)}]")
    print(f"    → Answer ↔ Retrieved chunks similarity (averaged)")
    print(f"    → High = answer is grounded in source material")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Loads embeddings, processes query, retrieves context,
    generates RAG response, then evaluates with RAGAS.
    """
    # Load embeddings
    logger.info("Loading chunk embeddings from '%s'...", EMBEDDINGS_FILE)
    try:
        df = joblib.load(EMBEDDINGS_FILE)
    except FileNotFoundError:
        logger.error(
            "'%s' not found. Please run Stage 3 (read_chunks.py) first.",
            EMBEDDINGS_FILE
        )
        return

    logger.info("Loaded %d chunks.", len(df))

    # Get user query
    query = input("Ask a question about the course: ").strip()
    if not query:
        logger.warning("Empty query received. Exiting.")
        return

    # Embed query
    logger.info("Embedding query...")
    try:
        query_embedding = create_embedding([query])[0]
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        return

    # Retrieve top chunks
    logger.info("Retrieving top %d relevant chunks...", TOP_K_RESULTS)
    top_chunks = retrieve_top_chunks(df, query_embedding, TOP_K_RESULTS)
    logger.info("Retrieved chunks from videos: %s", top_chunks["name"].tolist())

    # Build prompt and generate response
    prompt = build_prompt(query, top_chunks)
    logger.info("Generating response with '%s'...", GENERATION_MODEL)
    try:
        response = run_inference(prompt)
    except Exception as e:
        logger.error("LLM inference failed: %s", e)
        return

    # Print response
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")

    # Save response
    with open(RESPONSE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(response)
    logger.info("Response saved to '%s'.", RESPONSE_OUTPUT_FILE)

    # --- Cosine Similarity Evaluation ---
    chunk_texts = top_chunks["text"].tolist()
    scores = evaluate_with_cosine(query, response, chunk_texts)
    print_eval_scores(scores)
    save_eval_log(query, scores)


if __name__ == "__main__":
    main()