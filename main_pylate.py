import time
from dataclasses import replace
import numpy as np
import torch
from datasets import load_dataset
import logging
from pylate.models import ColBERT as PylateColBERT

from fde_generator import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

DATASET_REPO_ID = "zeta-alpha-ai/NanoFiQA2018"
COLBERT_MODEL_NAME = "ayushexel/colbert-ModernBERT-base-1-neg-1-epoch-gooaq-1995000" # Supports pylate models
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(f"Using device: {DEVICE}")


# --- Helper Functions ---
def load_nanobeir_dataset(repo_id: str) -> (dict, dict, dict):
    logging.info(f"Loading dataset from Hugging Face Hub: '{repo_id}'...")
    corpus_ds = load_dataset(repo_id, "corpus", split="train")
    queries_ds = load_dataset(repo_id, "queries", split="train")
    qrels_ds = load_dataset(repo_id, "qrels", split="train")

    corpus = {
        row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in corpus_ds
    }
    queries = {row["_id"]: row["text"] for row in queries_ds}
    qrels = {str(row["query-id"]): {str(row["corpus-id"]): 1} for row in qrels_ds}

    logging.info(f"Dataset loaded: {len(corpus)} documents, {len(queries)} queries.")
    return corpus, queries, qrels


def evaluate_recall(results: dict, qrels: dict, k: int) -> float:
    hits, total_queries = 0, 0
    for query_id, ranked_docs in results.items():
        relevant_docs = set(qrels.get(str(query_id), {}).keys())
        if not relevant_docs:
            continue
        total_queries += 1
        top_k_docs = set(list(ranked_docs.keys())[:k])
        if not relevant_docs.isdisjoint(top_k_docs):
            hits += 1
    return hits / total_queries if total_queries > 0 else 0.0


def to_numpy(tensor_or_array) -> np.ndarray:
    """Safely convert a PyTorch Tensor or a NumPy array to a float32 NumPy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")


class ColbertNativeRetriever:
    """Uses pylate's native ColBERT ranking (non-FDE)."""

    def __init__(self, model_name=COLBERT_MODEL_NAME):
        self.model = PylateColBERT(model_name_or_path=model_name, device=DEVICE)
        if hasattr(self.model[0].tokenizer, "model_max_length"): # For modernbert support
            self.model[0].tokenizer.model_input_names = ["input_ids", "attention_mask"]
        self.doc_embeddings_map = {}
        self.doc_ids = []

    def index(self, corpus: dict):
        self.doc_ids = list(corpus.keys())
        documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]
        doc_texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in documents_for_ranker]

        logging.info(
            f"[{self.__class__.__name__}] Generating ColBERT embeddings for all documents..."
        )
        doc_embeddings_list = self.model.encode(
            sentences=doc_texts,
            is_query=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.doc_embeddings_map = dict(zip(self.doc_ids, doc_embeddings_list))

    def search(self, query: str) -> dict:
        query_embedding = self.model.encode(
            sentences=query,
            is_query=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = {}
        with torch.no_grad():
            for doc_id, doc_embedding in self.doc_embeddings_map.items():
                late_interaction = torch.einsum("sh,th->st", query_embedding.to(DEVICE), doc_embedding.to(DEVICE))
                score = late_interaction.max(dim=1).values.sum()
                scores[doc_id] = score.item()

        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class ColbertFdeRetriever:
    """Uses a real ColBERT model to generate embeddings, then FDE for search."""

    def __init__(self, model_name=COLBERT_MODEL_NAME):
        self.model = PylateColBERT(model_name_or_path=model_name, device=DEVICE)
        if hasattr(self.model[0].tokenizer, "model_max_length"):
            self.model[0].tokenizer.model_input_names = ["input_ids", "attention_mask"]
        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=7,
            seed=42,
            fill_empty_partitions=True,  # Config for documents
        )
        self.fde_index, self.doc_ids = None, []

    def index(self, corpus: dict):
        self.doc_ids = list(corpus.keys())
        documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]
        doc_texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in documents_for_ranker]

        logging.info(f"[{self.__class__.__name__}] Generating native multi-vector embeddings...")
        doc_embeddings_list = self.model.encode(
            sentences=doc_texts,
            is_query=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logging.info(f"[{self.__class__.__name__}] Generating FDEs from ColBERT embeddings in BATCH mode...")
        self.fde_index = generate_document_fde_batch(doc_embeddings_list, self.doc_config)

    def search(self, query: str) -> dict:
        query_embeddings = self.model.encode(
            sentences=query,
            is_query=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_config = replace(self.doc_config, fill_empty_partitions=False)
        query_fde = generate_query_fde(query_embeddings, query_config)
        scores = self.fde_index @ query_fde
        return dict(sorted(zip(self.doc_ids, scores), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    logging.info("Initializing retrieval models...")
    retrievers = {
        "1. ColBERT (Native)": ColbertNativeRetriever(),
        "2. ColBERT + FDE": ColbertFdeRetriever(),
    }

    timings, final_results = {}, {}

    logging.info("--- PHASE 1: INDEXING ---")
    for name, retriever in retrievers.items():
        start_time = time.perf_counter()
        retriever.index(corpus)
        timings[name] = {"indexing_time": time.perf_counter() - start_time}
        logging.info(f"'{name}' indexing finished in {timings[name]['indexing_time']:.2f} seconds.")

    logging.info("--- PHASE 2: SEARCH & EVALUATION ---")
    for name, retriever in retrievers.items():
        logging.info(f"Running search for '{name}' on {len(queries)} queries...")
        query_times = []
        results = {}
        for query_id, query_text in queries.items():
            start_time = time.perf_counter()
            results[str(query_id)] = retriever.search(query_text)
            query_times.append(time.perf_counter() - start_time)

        timings[name]["avg_query_time"] = np.mean(query_times)
        final_results[name] = results
        logging.info(f"'{name}' search finished. Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")

    print("\n" + "=" * 85)
    print(f"{'FINAL REPORT':^85}")
    print(f"(Dataset: {DATASET_REPO_ID})")
    print("=" * 85)
    print(
        f"{'Retriever':<25} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10}"
    )
    print("-" * 85)

    for name in retrievers.keys():
        recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000

        print(
            f"{name:<25} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}"
        )

    print("=" * 85)
