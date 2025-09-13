import os
import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import networkx as nx
from langchain_ollama import ChatOllama
import pickle

PERSIST_DIR = "./chroma_persist"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 7

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

kg_path = os.path.join(PERSIST_DIR, "knowledge_graph.gpickle")
emb_path = os.path.join(PERSIST_DIR, "graph_embeddings.npy")
knowledge_graph, graph_embeddings, node_id_to_idx = None, None, None

if os.path.exists(kg_path) and os.path.exists(emb_path):
    with open(kg_path, 'rb') as f:
        knowledge_graph = pickle.load(f)
    graph_embeddings = np.load(emb_path)
    node_id_to_idx = {node_id: int(node_id.split('-')[1]) for node_id in knowledge_graph.nodes()}

def get_embedding(text):
    return embedder.encode(text)

def knowledge_graph_query(query, top_k=5):
    if knowledge_graph is None or graph_embeddings is None:
        return []
    query_embedding = get_embedding(query)
    similarities = []
    for node_id, idx in node_id_to_idx.items():
        node_embedding = graph_embeddings[idx]
        sim = np.dot(query_embedding, node_embedding)
        similarities.append((node_id, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node_id for node_id, _ in similarities[:top_k]]
    expanded_nodes = set(top_nodes)
    for node_id in top_nodes:
        neighbors = list(knowledge_graph.neighbors(node_id))
        expanded_nodes.update(neighbors)
    expanded_nodes = list(expanded_nodes)[:top_k * 3]
    retrieved_chunks = []
    for node_id in expanded_nodes:
        node_data = knowledge_graph.nodes[node_id]
        sim_score = next((sim for nid, sim in similarities if nid == node_id), 0.0)
        retrieved_chunks.append({
            "text": node_data["text"],
            "metadata": node_data["metadata"],
            "relevance": float(sim_score)
        })
    retrieved_chunks.sort(key=lambda x: x["relevance"], reverse=True)
    return retrieved_chunks[:top_k]

def hybrid_search(query, top_k=7):
    kg_results = []
    if knowledge_graph:
        kg_results = knowledge_graph_query(query, top_k=top_k)
    vector_results = []
    try:
        q_emb = get_embedding(query).tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            vector_results.append({
                "text": doc,
                "metadata": meta,
                "relevance": 1.0 - dist
            })
    except Exception:
        pass
    combined, seen_texts = [], set()
    all_results = kg_results + vector_results
    for res in all_results:
        if res["text"] not in seen_texts:
            combined.append(res)
            seen_texts.add(res["text"])
    combined.sort(key=lambda x: x["relevance"], reverse=True)
    return combined[:top_k]

def ask_rag_query(query, top_k=7):
    if collection.count() == 0:
        return "The document collection is empty. Please run the ingestion script first."
    results = hybrid_search(query, top_k=top_k)
    if not results:
        return "I could not find any relevant information in the documents to answer your question."
    context_blocks = []
    for i, res in enumerate(results):
        context_blocks.append(res['text'])
    context = "\n\n".join(context_blocks)
    prompt = f"""You are a helpful Q&A assistant. Your task is to answer the user's question based *only* on the context provided below.
Do not use any external knowledge. If the context does not contain the answer, you must state that you cannot answer based on the provided information.

CONTEXT:
---
{context}
---

QUESTION: {query}

ANSWER:"""
    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    query = os.environ.get("question", None)
    if not query:
        print("No question provided.")
        exit(1)
    answer = ask_rag_query(query, top_k=TOP_K)
    print(answer)
