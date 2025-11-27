import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from pathlib import Path

from petase_models import (
    AMINO_ACIDS,
    SurrogateModel,
    build_known_petase_index,
    EmbeddingCache,
    load_esm_embedder,
    make_one_hot_encoder,
)
from petase_scoring import (
    get_reference_sequence,
    hamming_distance,
    mutation_list,
    score_stability,
    set_reference_sequence,
)


def _maybe_get_esm_embedder(
    use_esm: bool,
    model_name: str,
    device: Optional[str],
    batch_size: int,
):
    """Return a cached ESM embedder if requested and available, else None."""
    if not use_esm:
        return None
    try:
        return load_esm_embedder(model_name=model_name, device=device, batch_size=batch_size)
    except ImportError:
        print(
            "ESM not installed; falling back to one-hot. Install with `pip install fair-esm torch`."
        )
        return None


# =========================
# 1. Basic sequence helpers
# =========================


# ==================================
# 2. PETase-specific constraints
# ==================================

# 1-based positions to freeze (PETase numbering)
FROZEN_POSITIONS = {
    160,
    206,
    237,  # catalytic triad
    87,
    161,
    185,  # oxyanion hole / cleft
    203,
    239,
    273,
    289,  # disulfides
}


def allowed_positions(seq: str) -> List[int]:
    """Return 0-based indices that are allowed to mutate."""
    L = len(seq)
    allowed = []
    for i in range(L):
        pos_1_based = i + 1
        if pos_1_based in FROZEN_POSITIONS:
            continue
        allowed.append(i)
    return allowed


def propose_single_mutants_guided(
    wt_seq: str, mut_list: Optional[List[Tuple[int, str]]] = None
) -> List[str]:
    """
    Generate single mutants guided by functional constraints and prior beneficial mutations.
    - mut_list: Optional list of (position, mutant_aa)
    """
    variants = []

    # Key known stabilizing mutations (from FAST-PETase and others)
    known_mutations = {
        121: ["E", "D"],  # S121E/D
        186: ["H"],  # D186H
        224: ["Q"],
        233: ["K"],
        280: ["A"],
        95: ["N"],
        201: ["I"],
        159: ["H"],
        229: ["Y"],
        181: ["A", "S"],  # P181A/S relieves beta-sheet distortion
    }

    if mut_list is None:
        for pos, aas in known_mutations.items():
            wt_aa = wt_seq[pos - 1]
            for aa in aas:
                if aa != wt_aa:
                    mutant = wt_seq[: pos - 1] + aa + wt_seq[pos:]
                    variants.append(mutant)
    else:
        for pos, aa in mut_list:
            wt_aa = wt_seq[pos - 1]
            if aa != wt_aa:
                mutant = wt_seq[: pos - 1] + aa + wt_seq[pos:]
                variants.append(mutant)

    return variants


def propose_double_mutants_guided(
    wt_seq: str,
    base_variants: List[str],
    max_variants: int = 500,
    max_dist: float = 0.2,
    embedding_provider: Optional[EmbeddingCache] = None,
) -> List[str]:
    """
    Generate double mutants by combining known beneficial positions and then
    filter them by ESM embedding distance to the wild-type.

    base_variants is kept for API compatibility but not used here.
    """
    variants = set()
    allowed = allowed_positions(wt_seq)
    # Positions we believe are often beneficial / interesting to mutate
    key_positions = list({121, 186, 224, 233, 280, 95, 201, 159, 229, 181} & set(allowed))

    # Oversample candidates before filtering in embedding space
    oversample_target = max_variants * 5

    for i in range(len(key_positions)):
        for j in range(i + 1, len(key_positions)):
            pi, pj = key_positions[i], key_positions[j]
            for aai in AMINO_ACIDS:
                if aai == wt_seq[pi - 1]:
                    continue
                for aaj in AMINO_ACIDS:
                    if aaj == wt_seq[pj - 1]:
                        continue
                    seq = list(wt_seq)
                    seq[pi - 1] = aai
                    seq[pj - 1] = aaj
                    variants.add("".join(seq))
                    if len(variants) >= oversample_target:
                        break
                if len(variants) >= oversample_target:
                    break
            if len(variants) >= oversample_target:
                break
        if len(variants) >= oversample_target:
            break

    if not variants:
        return []

    if embedding_provider is None:
        return list(variants)[:max_variants]

    # ESM embedding filter: keep only those close to WT in latent space
    all_seqs = [wt_seq] + list(variants)
    all_embs = embedding_provider.get(all_seqs)
    wt_emb = all_embs[0].reshape(1, -1)
    var_embs = all_embs[1:]

    dists = cosine_distances(wt_emb, var_embs)[0]
    filtered = [s for s, d in zip(list(variants), dists) if d <= max_dist]
    print(f"Filtered {len(filtered)} double mutants within {max_dist} embedding distance of WT")

    return filtered[:max_variants]


# ==================================
# 3. Simple QD archive + acquisition
# ==================================


class QDArchive:
    """
    Very simple QD archive over (mutation_count, stability_bin).
    """

    def __init__(self, bin_width: float = 0.5):
        self.bin_width = bin_width
        self.cells = {}  # (mut_count, stab_bin) -> (seq, score)

    def _key(self, seq: str, stability_score: float) -> Tuple[int, int]:
        wt_seq = get_reference_sequence()
        if not wt_seq:
            raise ValueError("Reference sequence not set; call set_reference_sequence first.")
        mut_count = hamming_distance(seq, wt_seq)
        stab_bin = int(stability_score / self.bin_width)
        return mut_count, stab_bin

    def insert(self, seq: str, stability_score: float):
        key = self._key(seq, stability_score)
        if key not in self.cells or stability_score > self.cells[key][1]:
            self.cells[key] = (seq, stability_score)

    def elites(self) -> List[str]:
        return [v[0] for v in self.cells.values()]


def acquisition_ucb(mean: np.ndarray, std: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """UCB = mean + beta * std; higher is better."""
    return mean + beta * std


def pick_diverse_batch(
    candidates: List[str],
    scores: np.ndarray,
    batch_size: int,
    min_hamming: int = 3,  # kept for compatibility, not used
    min_embedding_dist: float = 0.05,
    embedding_provider: Optional[EmbeddingCache] = None,
) -> List[str]:
    """
    Greedy diversity filter. Uses ESM embedding distance when a provider is available,
    otherwise falls back to simple Hamming-distance diversity.
    """
    if not candidates:
        return []

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    if embedding_provider is None:
        selected: List[str] = []
        for seq, _ in ranked:
            if not selected or all(hamming_distance(seq, s) >= min_hamming for s in selected):
                selected.append(seq)
                if len(selected) >= batch_size:
                    break
        return selected

    embeddings = embedding_provider.get(candidates)
    # align embeddings with ranked order
    emb_map = {seq: emb for seq, emb in zip(candidates, embeddings)}

    selected: List[str] = []
    selected_embs: List[np.ndarray] = []
    for seq, _ in ranked:
        emb = emb_map[seq]
        if not selected:
            selected.append(seq)
            selected_embs.append(emb)
            if len(selected) >= batch_size:
                break
            continue

        dists = cosine_distances(emb.reshape(1, -1), np.vstack(selected_embs))[0]
        if all(d >= min_embedding_dist for d in dists):
            selected.append(seq)
            selected_embs.append(emb)
            if len(selected) >= batch_size:
                break

    return selected


# ==================================
# 4. Main DBTL-like loop
# ==================================


def initial_round(
    wt_seq: str,
    n_single: int = 200,
    n_double: int = 500,
    use_esm: bool = False,
    esm_model: str = "esm2_t6_8M_UR50D",
    esm_batch_size: int = 8,
    esm_device: Optional[str] = None,
    embedding_provider: Optional[EmbeddingCache] = None,
):
    """
    Generate initial variants, score them with stability, and fit surrogate.
    Returns: surrogate model, archive, training data.
    """
    set_reference_sequence(wt_seq)
    # 1) Initial variant set
    singles = propose_single_mutants_guided(wt_seq)
    doubles = propose_double_mutants_guided(
        wt_seq,
        singles,
        max_variants=n_double,
        embedding_provider=embedding_provider if use_esm else None,
    )
    seqs = [wt_seq] + singles + doubles

    # 2) Score stability (and optionally filter by activity floor)
    stability_scores = []
    for s in seqs:
        stability_scores.append(score_stability(s))

    # 3) Fit surrogate
    encoder = (
        embedding_provider.get
        if (use_esm and embedding_provider is not None)
        else make_one_hot_encoder()
    )
    if use_esm and embedding_provider is not None:
        print(f"Using ESM embeddings ({esm_model}) for surrogate features")

    surrogate = SurrogateModel(encoder=encoder)
    surrogate.fit(seqs, stability_scores)

    # 4) Build initial QD archive
    archive = QDArchive()
    for s, stab in zip(seqs, stability_scores):
        archive.insert(s, stab)

    return surrogate, archive, seqs, stability_scores


def propose_new_candidates(base_seqs: List[str], n_candidates: int = 500) -> List[str]:
    """Mutate around current elites to propose new candidates."""
    candidates = set()
    allowed = None  # lazily compute

    while len(candidates) < n_candidates:
        parent = random.choice(base_seqs)
        if allowed is None:
            allowed = allowed_positions(parent)
        seq_list = list(parent)
        # choose 1–3 mutation sites
        k = random.choice([1, 2, 3])
        pos_to_mutate = random.sample(allowed, k=k)
        for idx in pos_to_mutate:
            wt_aa = seq_list[idx]
            aa_choices = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            seq_list[idx] = random.choice(aa_choices)
        child = "".join(seq_list)
        # enforce hard constraints (triad, disulfides) implicitly via allowed_positions
        candidates.add(child)

    return list(candidates)


def active_learning_round(
    surrogate: SurrogateModel,
    archive: QDArchive,
    wt_seq: str,
    n_candidates: int = 2000,
    batch_for_lab: int = 16,
    beta: float = 1.0,
    embedding_provider: Optional[EmbeddingCache] = None,
) -> Dict:
    """
    One AL round:
      - propose candidates around current elites
      - score surrogate mean/std
      - compute UCB
      - pick diverse top batch for lab
      - also score chosen batch with true stability proxy for retraining
    """
    # 1) Propose around QD elites (or WT if archive small)
    elites = archive.elites() or [wt_seq]
    candidates = propose_new_candidates(elites, n_candidates=n_candidates)

    # 2) Surrogate predictions
    mean, std = surrogate.predict_with_uncertainty(candidates)
    ucb = acquisition_ucb(mean, std, beta=beta)

    # 3) Rank by UCB and pick diverse batch
    batch = pick_diverse_batch(
        candidates,
        ucb,
        batch_size=batch_for_lab,
        embedding_provider=embedding_provider,
    )

    # 4) Score chosen batch with "true" (proxy) stability
    stab_batch = [score_stability(s) for s in batch]

    # 5) Update surrogate training set and refit
    # (In reality, you'd accumulate all data across rounds)
    surrogate.fit(batch, stab_batch)

    # 6) Update archive with new real scores
    for s, stab in zip(batch, stab_batch):
        archive.insert(s, stab)

    return {
        "batch_sequences": batch,
        "batch_stabilities": stab_batch,
        "ucb_scores": [ucb[candidates.index(s)] for s in batch],
    }


# ==================================
# 7. Putting it together
# ==================================


def run_pipeline(
    wt_seq: str,
    n_rounds: int = 3,
    initial_n_single: int = 200,
    initial_n_double: int = 500,
    n_candidates_per_round: int = 2000,
    batch_for_lab: int = 16,
    use_esm: bool = False,
    esm_model: str = "esm2_t6_8M_UR50D",
    esm_batch_size: int = 8,
    esm_device: Optional[str] = None,
    embedding_cache_path: Optional[str] = None,
):
    set_reference_sequence(wt_seq)
    embedder_fn = _maybe_get_esm_embedder(use_esm, esm_model, esm_device, esm_batch_size)
    embedding_provider = None
    cache_path = None
    if embedder_fn is not None:
        default_cache = Path("data/processed") / f"esm_cache_{esm_model}.npz"
        cache_path = Path(embedding_cache_path) if embedding_cache_path else default_cache
        embedding_provider = EmbeddingCache(embedder_fn, str(cache_path))

    esm_active = use_esm and embedding_provider is not None

    try:
        if esm_active:
            # Build ESM index of known PETase variants (currently just WT)
            build_known_petase_index(
                wt_seq,
                model_name=esm_model,
                device=esm_device,
                batch_size=esm_batch_size,
                embedder=embedding_provider.get,
            )

        print("Running initial round...")
        surrogate, archive, train_seqs, train_stab = initial_round(
            wt_seq,
            n_single=initial_n_single,
            n_double=initial_n_double,
            use_esm=esm_active,
            esm_model=esm_model,
            esm_batch_size=esm_batch_size,
            esm_device=esm_device,
            embedding_provider=embedding_provider if esm_active else None,
        )
        print(f"Initial training set size: {len(train_seqs)}")
        print(f"Initial stability score range: {min(train_stab):.3f} to {max(train_stab):.3f}")

        for r in range(1, n_rounds + 1):
            print(f"\n=== Active Learning Round {r} ===")
            result = active_learning_round(
                surrogate,
                archive,
                wt_seq,
                n_candidates=n_candidates_per_round,
                batch_for_lab=batch_for_lab,
                beta=max(0.5, 2.0 / (r + 1)),  # anneal exploration
                embedding_provider=embedding_provider if esm_active else None,
            )

            print("Selected batch for lab testing:")
            for s, stab in zip(result["batch_sequences"], result["batch_stabilities"]):
                muts = mutation_list(wt_seq, s)
                print(f" Sequence: {s}\n Mutations: {muts}\n Stability proxy: {stab:.3f}\n")

        # At the end you have:
        # - archive.elites(): diverse high-stability variants
        # - last selected batch: good lab candidates
        return archive
    finally:
        if embedding_provider is not None:
            embedding_provider.save()


if __name__ == "__main__":
    # Example usage with a real PETase WT sequence
    wt_example = (
        "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKW"
        "WGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFS"
        "SVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
    )
    # Set use_esm=True to encode variants with ESM embeddings instead of one-hot.
    # Requires `pip install fair-esm torch` beforehand.
    run_pipeline(
        wt_example,
        n_rounds=2,
        use_esm=True,
        esm_model="esm2_t6_8M_UR50D",
        esm_batch_size=8,
    )
