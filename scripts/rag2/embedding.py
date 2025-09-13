import os
import pickle
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from tqdm import tqdm

PERSIST_DIR = "./chroma_persist"
COLLECTION_NAME = "rag_collection"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKED_PATH = "chunked_data.pkl"

os.makedirs(PERSIST_DIR, exist_ok=True)

# --- Load chunked data ---
with open(CHUNKED_PATH, "rb") as f:
    chunked = pickle.load(f)
chunked_texts = chunked["texts"]
chunked_metadatas = chunked["metadatas"]

print(f"Loaded {len(chunked_texts)} chunks for embedding.")

# --- Embedding ---
embedder = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = []
for chunk in tqdm(chunked_texts, desc="Embedding"):
    emb = embedder.encode(chunk)
    embeddings.append(emb.tolist())

# --- Store in ChromaDB ---
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("Loaded existing collection.")
except Exception:
    collection = chroma_client.create_collection(COLLECTION_NAME)
    print("Created new collection.")

try:
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        print(f"Deleted {len(all_ids)} old chunks from the collection.")
except IndexError:
    print("Collection is already empty.")

collection.add(
    documents=chunked_texts,
    embeddings=embeddings,
    metadatas=chunked_metadatas,
    ids=[f"chunk-{i}" for i in range(len(chunked_texts))]
)
print(f"Indexed {len(chunked_texts)} chunks in ChromaDB.")

# --- Build knowledge graph ---
G = nx.Graph()
for i, (chunk, meta) in enumerate(zip(chunked_texts, chunked_metadatas)):
    node_id = f"chunk-{i}"
    G.add_node(node_id, text=chunk, metadata=meta)
for idx, meta in enumerate(chunked_metadatas):
    if idx > 0 and meta['source'] == chunked_metadatas[idx-1]['source']:
        prev_id = f"chunk-{idx-1}"
        curr_id = f"chunk-{idx}"
        G.add_edge(prev_id, curr_id, relation="next_chunk")

graph_path = os.path.join(PERSIST_DIR, "knowledge_graph.gpickle")
with open(graph_path, 'wb') as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

np.save(os.path.join(PERSIST_DIR, "graph_embeddings.npy"), np.array(embeddings, dtype=np.float32))
print("Saved knowledge graph and node embeddings.")
