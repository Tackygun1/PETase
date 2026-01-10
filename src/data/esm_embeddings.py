"""
src/data/esm_embeddings.py

Generates ESM-2 embeddings for protein sequences.
Outputs are saved as numpy arrays or pickle dictionaries.

Supports batching, GPU acceleration, and resuming.

WARNING: Requires significant memory for large datasets. (Only run on cuda-enabled machines with sufficient VRAM.)
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class ESMEmbedder:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | None = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading ESM model ({self.model_name}) on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_sequences(self, sequences: list[str], ids: list[str]) -> dict[str, np.ndarray]:
        """
        Compute per-sequence embeddings (mean pooled).
        Returns dict {id: embedding (np.ndarray)}.
        """
        embeddings = {}
        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Embedding batches"):
            batch_seqs = sequences[i : i + self.batch_size]
            batch_ids = ids[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1).cpu().numpy()
            for pid, emb in zip(batch_ids, pooled):
                embeddings[pid] = emb
        return embeddings

    def save_embeddings(self, embeddings: dict[str, np.ndarray], out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **embeddings)
        print(f" Saved embeddings to {out_path}")


def main(
    input_csv="data/processed/clean_sequences.csv", output_path="data/processed/esm_embeddings.npz"
):
    df = pd.read_csv(input_csv)
    ids, seqs = df["id"].tolist(), df["sequence"].tolist()
    embedder = ESMEmbedder()
    embeddings = embedder.embed_sequences(seqs, ids)
    embedder.save_embeddings(embeddings, Path(output_path))


if __name__ == "__main__":
    main()
