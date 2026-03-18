from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import requests
import logging
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "chunks_with_embeddings.joblib"

OLLAMA_EMBED_URL    = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

COURSE_SUBJECT = "Data Analysis using Python"

df = joblib.load(EMBEDDINGS_FILE)



def format_timestamp(sec):
    return f"{int(sec//60):02}:{int(sec%60):02}"

def embed(text):
    """Embeds a single string using BGE-M3 via Ollama."""
    r = requests.post(OLLAMA_EMBED_URL, json={"model": "bge-m3", "input": [text]})
    return r.json()["embeddings"][0]

def embed_many(texts: list) -> list:
    """Embeds a list of strings in one Ollama call."""
    r = requests.post(OLLAMA_EMBED_URL, json={"model": "bge-m3", "input": texts})
    return r.json()["embeddings"]

def retrieve(query_vec, top_k=5):
    matrix = np.vstack(df["embedding"].values)
    sims   = cosine_similarity(matrix, [query_vec]).flatten()
    idx    = sims.argsort()[::-1][:top_k]
    result = df.loc[idx].copy()
    result["similarity"] = sims[idx].tolist()
    result["start"] = result["start"].apply(format_timestamp)
    result["end"]   = result["end"].apply(format_timestamp)
    result["text"]  = result["text"].str[:400]
    return result

def generate(prompt):
    r = requests.post(OLLAMA_GENERATE_URL, json={
        "model": "llama3.2", "prompt": prompt, "stream": False
    })
    return r.json()["response"]

# Cosine similarity evaluation

def evaluate_with_cosine(question: str, answer: str, chunk_texts: list) -> dict:
    """
    Evaluates answer quality using cosine similarity on BGE-M3 embeddings.

    Two metrics:

    answer_relevancy:
    How semantically close is the generated answer to the original question?
    High = answer is on-topic. Low = answer is vague or off-topic.

    faithfulness :
    How closely does the answer language match the retrieved source chunks?

    range 0.0 – 1.0.
    """
    try:

        # Embed question and answer
        question_vec = np.array(embed(question))
        answer_vec   = np.array(embed(answer))

        # answer_relevancy: 
        answer_relevancy = float(
            cosine_similarity([question_vec], [answer_vec])[0][0]
        )

        # faithfulness: 
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


# Routes

@app.route("/query", methods=["POST"])
def query():
    """
    POST { "question": "..." }

    Returns:
    {
      "answer": "...",
      "evaluation": {
        "answer_relevancy": 0.87,   // question ↔ answer similarity  (0–1)
        "faithfulness":     0.91    // answer   ↔ chunks similarity   (0–1)
      }
    }

    """
    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # --- Retrieve ---
    query_vec  = embed(question)
    top_chunks = retrieve(query_vec)

    # --- Generate ---
    context = top_chunks[["name", "number", "start", "end", "text"]].to_json(orient="records")
    prompt  = f"""You are a teaching assistant for a course on {COURSE_SUBJECT}.
Here are transcript chunks from lecture videos (name, number, start, end, text):
{context}
---
Question: "{question}"
Answer naturally, mention which video and timestamp covers this topic. Use "minutes and seconds" format when citing timestamps (e.g. "at 5 minutes 12 seconds"). If the answer is not in the provided chunks, say "Sorry, I don't know". Be concise and to the point."""

    answer = generate(prompt)

    # --- Cosine similarity evaluation ---
    chunk_texts = top_chunks["text"].tolist()
    eval_scores = evaluate_with_cosine(question, answer, chunk_texts)

    return jsonify({
        "answer":     answer,
        "evaluation": eval_scores,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)