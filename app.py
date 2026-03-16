from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

EMBEDDINGS_FILE = "chunks_with_embeddings.joblib"

OLLAMA_EMBED_URL   = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


COURSE_SUBJECT = "Data Analysis using Python"

df = joblib.load(EMBEDDINGS_FILE)



def format_timestamp(sec):
    return f"{int(sec//60):02}:{int(sec%60):02}"

def embed(text):
    r = requests.post(OLLAMA_EMBED_URL, json={"model": "bge-m3", "input": [text]})
    return r.json()["embeddings"][0]

def retrieve(query_vec, top_k=5):
    matrix = np.vstack(df["embedding"].values)
    sims = cosine_similarity(matrix, [query_vec]).flatten()
    idx = sims.argsort()[::-1][:top_k]
    result = df.loc[idx].copy()
    result["similarity"] = sims[idx].tolist()
    result["start"] = result["start"].apply(format_timestamp)
    result["end"]   = result["end"].apply(format_timestamp)
    result["text"]  = result["text"].str[:400] # limit text length for output
    return result

def generate(prompt):
    r = requests.post(OLLAMA_GENERATE_URL, json={
        "model": "llama3.2", "prompt": prompt, "stream": False
    })
    return r.json()["response"]

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    query_vec   = embed(question)
    top_chunks  = retrieve(query_vec)

    context = top_chunks[["name","number","start","end","text"]].to_json(orient="records")
    prompt = f"""You are a teaching assistant for a course on {COURSE_SUBJECT}.
Here are transcript chunks from lecture videos (name, number, start, end, text):
{context}
---
Question: "{question}"
Answer naturally, mention which video and timestamp covers this topic. Use "minutes and seconds" format when citing timestamps (e.g. "at 5 minutes 12 seconds").If the answer is not in the provided chunks, say "Sorry, I don't know". Be concise and to the point"""

    answer = generate(prompt)

    # # Format chunks for output
    # chunks_out = top_chunks[["name","number","start","end","text","similarity"]].to_dict(orient="records")
    return jsonify({"answer": answer})

# Metrics computed:
#
#   MRR  (Mean Reciprocal Rank)
#        Measures how high the first relevant result appears in the ranked
#        list. MRR = 1.0 means the correct chunk is always ranked #1.
#        MRR = 0.5 means it's on average ranked #2.
#        Formula: MRR = (1/N) * sum(1 / rank_of_first_relevant)
#
#   Precision@K
#        Fraction of the top-K retrieved chunks that are actually relevant.
#        e.g. Precision@5 = 0.8 means 4 of the 5 chunks matched the query.
#        Formula: P@K = (relevant chunks in top-K) / K
#
# POST body format:
#   {
#     "tests": [
#       {
#         "query": "How does groupby work?",
#         "relevant_keywords": ["groupby", "aggregate", "split"]
#       },
#       ...
#     ],
#     "top_k": 5
#   }
#
# A chunk is considered relevant if its text contains ANY of the keywords.
# ---------------------------------------------------------------------------

@app.route("/evaluate", methods=["POST"])
def evaluate():
    body  = request.json or {}
    tests = body.get("tests", [])
    top_k = int(body.get("top_k", 5))

    if not tests:
        return jsonify({"error": "Provide a 'tests' list with query + relevant_keywords"}), 400

    matrix = np.vstack(df["embedding"].values)

    reciprocal_ranks  = []
    precisions_at_k   = []
    per_query_results = []

    for test in tests:
        query    = test.get("query", "").strip()
        keywords = [kw.lower() for kw in test.get("relevant_keywords", [])]

        if not query:
            continue

        q_vec     = embed(query)
        sims      = cosine_similarity(matrix, [q_vec]).flatten()
        idx       = sims.argsort()[::-1][:top_k]
        top_texts = df.loc[idx, "text"].str.lower().tolist()

        # A chunk is relevant if it contains any keyword
        relevance = [any(kw in text for kw in keywords) for text in top_texts]

        # MRR — find rank of first relevant result
        rr = 0.0
        for rank, is_rel in enumerate(relevance, start=1):
            if is_rel:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        precision = sum(relevance) / top_k
        precisions_at_k.append(precision)

        per_query_results.append({
            "query":           query,
            "reciprocal_rank": round(rr, 4),
            "precision_at_k":  round(precision, 4),
            "relevant_count":  sum(relevance),
            "top_k":           top_k,
        })

    n = len(reciprocal_ranks)
    return jsonify({
        "num_queries":    n,
        "top_k":          top_k,
        "mrr":            round(float(np.mean(reciprocal_ranks)), 4),
        "precision_at_k": round(float(np.mean(precisions_at_k)), 4),
        "per_query":      per_query_results,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)