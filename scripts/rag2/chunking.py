import os
import glob
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "output2"
CHUNKED_PATH = "chunked_data.pkl"

file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
documents, metadatas = [], []
for path in file_paths:
    with open(path, encoding='utf-8') as f:
        text = f.read()
        documents.append(text)
        metadatas.append({"source": os.path.basename(path)})

if not documents:
    print(f"No .txt files found in {DATA_DIR}. Please add files.")
    exit(1)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_texts, chunked_metadatas = [], []
for doc, meta in zip(documents, metadatas):
    chunks = splitter.split_text(doc)
    for i, chunk in enumerate(chunks):
        chunked_texts.append(chunk)
        chunked_meta = meta.copy()
        chunked_meta["chunk_id"] = i
        chunked_metadatas.append(chunked_meta)

print(f"Total chunks produced: {len(chunked_texts)}")

# Save to disk as one file (fast, portable)
with open(CHUNKED_PATH, "wb") as f:
    pickle.dump({"texts": chunked_texts, "metadatas": chunked_metadatas}, f)

# --- FIX: Removed Unicode character for compatibility ---
print(f"Chunked data saved to {CHUNKED_PATH}")
