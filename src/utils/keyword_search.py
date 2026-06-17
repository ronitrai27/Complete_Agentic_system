import os
import json
import pickle
from typing import List, Dict, Any
from loguru import logger
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback to hardcoded standard English stopwords if NLTK data lookup fails
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
        "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now"
    }

def tokenize(text: str) -> List[str]:
    """
    Tokenize input text for BM25.
    Lowercases and splits by word characters.
    """
    return [word.strip(",.?!()[]{}:;\"'").lower() for word in text.split() if word.strip()]

class BM25IndexManager:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = [] # Each item is {"text": str, "metadata": dict}
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None
        self.load()

    def load(self):
        """Loads index from disk if it exists."""
        if os.path.exists(INDEX_PATH):
            try:
                with open(INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    if self.tokenized_corpus:
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks from disk.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}. Starting fresh.")
                self.chunks = []
                self.tokenized_corpus = []
                self.bm25 = None
        else:
            logger.info("No BM25 index found. Starting with empty index.")

    def save(self):
        """Saves the index to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        try:
            with open(INDEX_PATH, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "tokenized_corpus": self.tokenized_corpus
                }, f)
            logger.info(f"Saved BM25 index with {len(self.chunks)} chunks to {INDEX_PATH}.")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def add_chunks(self, document_id: str, new_chunks: List[str], metadata_base: Dict[str, Any] = None):
        """
        Adds new chunks to the BM25 index.
        """
        logger.info(f"Adding {len(new_chunks)} chunks for {document_id} to BM25 index...")
        
        # Remove any existing chunks for this document_id to prevent duplicate indexing
        self.chunks = [c for c in self.chunks if c.get("metadata", {}).get("document_id") != document_id]
        
        for i, chunk in enumerate(new_chunks):
            metadata = (metadata_base or {}).copy()
            metadata["document_id"] = document_id
            metadata["chunk_index"] = i
            
            self.chunks.append({
                "text": chunk,
                "metadata": metadata
            })
            
        # Rebuild tokenized corpus and BM25 index
        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
            
        self.save()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the BM25 index.
        """
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty.")
            return []

        tokenized_query = tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair each chunk with its score and index
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            score = float(scores[i])
            if score >= 0: # Include all non-negative scoring chunks (BM25 can return 0 for small corpora)
                scored_chunks.append((score, chunk))
                
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, chunk in scored_chunks[:top_k]:
            results.append({
                "text": chunk["text"],
                "score": score,
                "metadata": chunk["metadata"]
            })
            
        return results

# Singleton manager
_manager = None

def get_bm25_manager() -> BM25IndexManager:
    global _manager
    if _manager is None:
        _manager = BM25IndexManager()
    return _manager

def add_to_bm25_index(document_id: str, chunks: List[str], metadata: Dict[str, Any] = None):
    manager = get_bm25_manager()
    manager.add_chunks(document_id, chunks, metadata)

def query_bm25(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    manager = get_bm25_manager()
    return manager.query(query_text, top_k)
