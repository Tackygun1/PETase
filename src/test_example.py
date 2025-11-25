import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from unittest import result

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# =========================
# 1. Basic sequence helpers
# =========================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def one_hot_encode(seq: str) -> np.ndarray:
    """Simple one-hot per position; returns (L, 20) flattened to (L*20,)."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    L = len(seq)
    mat = np.zeros((L, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            mat[i, aa_to_idx[aa]] = 1.0
    return mat.reshape(-1)


def hamming_distance(a: str, b: str) -> int:
    assert len(a) == len(b)
    return sum(x != y for x, y in zip(a, b))


def mutation_list(parent: str, child: str) -> List[str]:
    """Return mutations as e.g. ['S121E', 'D186H']."""
    muts = []
    for i, (aa0, aa1) in enumerate(zip(parent, child), start=1):
        if aa0 != aa1:
            muts.append(f"{aa0}{i}{aa1}")
    return muts


# ==================================
# 2. PETase-specific constraints
# ==================================

# 1-based positions to freeze (PETase numbering)
FROZEN_POSITIONS = {
    160, 206, 237,  # catalytic triad
    87, 161, 185,   # oxyanion hole / cleft
    203, 239, 273, 289  # disulfides
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


def propose_single_mutants_guided(wt_seq: str, mut_list: Optional[List[Tuple[int, str]]] = None) -> List[str]:
    """
    Generate single mutants guided by functional constraints and prior beneficial mutations.
    - mut_list: Optional list of (position, mutant_aa)
    """
    variants = []
    allowed = allowed_positions(wt_seq)

    # Key known stabilizing mutations (from FAST-PETase and others)
    known_mutations = {
        121: ['E', 'D'],  # S121E/D
        186: ['H'],       # D186H
        224: ['Q'],
        233: ['K'],
        280: ['A'],
        95: ['N'],
        201: ['I'],
        159: ['H'],
        229: ['Y'],
        181: ['A', 'S']   # P181A/S relieves beta-sheet distortion
    }

    if mut_list is None:
        for pos, aas in known_mutations.items():
            wt_aa = wt_seq[pos - 1]
            for aa in aas:
                if aa != wt_aa:
                    mutant = wt_seq[:pos - 1] + aa + wt_seq[pos:]
                    variants.append(mutant)
    else:
        for pos, aa in mut_list:
            wt_aa = wt_seq[pos - 1]
            if aa != wt_aa:
                mutant = wt_seq[:pos - 1] + aa + wt_seq[pos:]
                variants.append(mutant)

    return variants

def propose_double_mutants_guided(wt_seq: str, base_variants: List[str]) -> List[str]:
    """
    Generate double mutants by combining known beneficial mutations,
    or mutating two loop/surface-exposed residues excluding critical positions.
    """
    variants = set()
    allowed = allowed_positions(wt_seq)
    key_positions = list({121, 186, 224, 233, 280, 95, 201, 159, 229, 181} & set(allowed))

    for i in range(len(key_positions)):
        for j in range(i + 1, len(key_positions)):
            pi, pj = key_positions[i], key_positions[j]
            for aai in AMINO_ACIDS:
                for aaj in AMINO_ACIDS:
                    if aai != wt_seq[pi - 1] and aaj != wt_seq[pj - 1]:
                        seq = list(wt_seq)
                        seq[pi - 1] = aai
                        seq[pj - 1] = aaj
                        variants.add("".join(seq))
                        if len(variants) >= 500:
                            return list(variants)
    return list(variants)



# ==================================
# 3. Scoring: stability & activity
# ==================================

def score_stability(seq: str) -> float:
    """
    Placeholder. Replace with:
      - FoldX / Rosetta ΔΔG call (negative = better)
      - or ML stability predictor
    Convention here: HIGHER = BETTER (so you might invert ΔΔG).
    """
    # Dummy: penalize number of mutations and random noise
    mut_count = hamming_distance(seq, GLOBAL_WT_SEQUENCE)
    return -mut_count + np.random.normal(scale=0.2)


def score_activity(seq: str) -> Optional[float]:
    """
    Placeholder. Replace with:
      - Docking score with PET fragments
      - ML activity proxy
      - or lab activity if available
    Return None if you don't want to use activity yet.
    """
    return None


# ==================================
# 4. Surrogate model + embeddings
# ==================================

def make_one_hot_encoder():
    """Return a callable that encodes a list of sequences to one-hot vectors."""
    def _encode(seqs: List[str]) -> np.ndarray:
        return np.vstack([one_hot_encode(s) for s in seqs])
    return _encode


def load_esm_embedder(
    model_name: str = "esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    batch_size: int = 8,
):
    """
    Lazily load an ESM model and return a callable that embeds sequences to vectors.

    Requires:
        pip install fair-esm torch
    """
    import torch  # imported lazily to keep baseline dependencies light
    import esm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    layer = model.num_layers

    def _encode(seqs: List[str]) -> np.ndarray:
        data = [(str(i), s) for i, s in enumerate(seqs)]
        outputs = []
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            _, batch_strs, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            with torch.no_grad():
                res = model(tokens, repr_layers=[layer], return_contacts=False)
            reps = res["representations"][layer]
            for i, seq in enumerate(batch_strs):
                seq_len = len(seq)
                emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy()  # mean-pooled residue reps
                outputs.append(emb)
        return np.stack(outputs, axis=0)

    return _encode

class SurrogateModel:
    def __init__(self, encoder=None):
        # Random forest gives some notion of uncertainty via per-tree variance
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
        self._is_fitted = False
        self._encode = encoder or make_one_hot_encoder()

    def fit(self, seqs: List[str], scores: List[float]):
        X = self._encode(seqs)
        y = np.array(scores, dtype=float)
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_with_uncertainty(self, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_fitted, "Surrogate not fitted yet"
        X = self._encode(seqs)
        # Use per-tree predictions to estimate variance
        all_tree_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=1)
        mean = all_tree_preds.mean(axis=1)
        std = all_tree_preds.std(axis=1)
        return mean, std


# ==================================
# 5. Simple QD archive + acquisition
# ==================================

class QDArchive:
    """
    Very simple QD archive over (mutation_count, stability_bin).
    """
    def __init__(self, bin_width: float = 0.5):
        self.bin_width = bin_width
        self.cells = {}  # (mut_count, stab_bin) -> (seq, score)

    def _key(self, seq: str, stability_score: float) -> Tuple[int, int]:
        mut_count = hamming_distance(seq, GLOBAL_WT_SEQUENCE)
        stab_bin = int(stability_score / self.bin_width)
        return mut_count, stab_bin

    def insert(self, seq: str, stability_score: float):
        key = self._key(seq, stability_score)
        if key not in self.cells or stability_score > self.cells[key][1]:
            self.cells[key] = (seq, stability_score)

    def elites(self) -> List[str]:
        return [v[0] for v in self.cells.values()]


def acquisition_ucb(
    mean: np.ndarray, std: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """UCB = mean + beta * std; higher is better."""
    return mean + beta * std


def pick_diverse_batch(
    candidates: List[str],
    scores: np.ndarray,
    batch_size: int,
    min_hamming: int = 3
) -> List[str]:
    """Greedy diversity filter based on Hamming distance."""
    selected = []
    # sort candidates by score descending
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    for seq, _ in ranked:
        if len(selected) == 0:
            selected.append(seq)
            if len(selected) >= batch_size:
                break
            continue
        if all(hamming_distance(seq, s) >= min_hamming for s in selected):
            selected.append(seq)
            if len(selected) >= batch_size:
                break
    return selected


# ==================================
# 6. Main DBTL-like loop
# ==================================

GLOBAL_WT_SEQUENCE = ""  # will be set in main()

def initial_round(
    wt_seq: str,
    n_single: int = 200,
    n_double: int = 500,
    use_esm: bool = False,
    esm_model: str = "esm2_t6_8M_UR50D",
    esm_batch_size: int = 8,
    esm_device: Optional[str] = None,
):
    """
    Generate initial variants, score them with stability, and fit surrogate.
    Returns: surrogate model, archive, training data.
    """
    # 1) Initial variant set
    singles = propose_single_mutants_guided(wt_seq)
    doubles = propose_double_mutants_guided(wt_seq, singles)
    seqs = [wt_seq] + singles + doubles

    # 2) Score stability (and optionally filter by activity floor)
    stability_scores = []
    for s in seqs:
        stability_scores.append(score_stability(s))

    # 3) Fit surrogate
    encoder = make_one_hot_encoder()
    if use_esm:
        try:
            encoder = load_esm_embedder(
                model_name=esm_model,
                device=esm_device,
                batch_size=esm_batch_size,
            )
            print(f"Using ESM embeddings ({esm_model}) for surrogate features")
        except ImportError:
            print("ESM not installed; falling back to one-hot. Install with `pip install fair-esm torch`.")
            encoder = make_one_hot_encoder()

    surrogate = SurrogateModel(encoder=encoder)
    surrogate.fit(seqs, stability_scores)

    # 4) Build initial QD archive
    archive = QDArchive()
    for s, stab in zip(seqs, stability_scores):
        archive.insert(s, stab)

    return surrogate, archive, seqs, stability_scores


def propose_new_candidates(
    base_seqs: List[str],
    n_candidates: int = 500
) -> List[str]:
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
    beta: float = 1.0
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
    batch = pick_diverse_batch(candidates, ucb, batch_size=batch_for_lab)

    # 4) Score chosen batch with "true" (proxy) stability
    stab_batch = [score_stability(s) for s in batch]

    # 5) Update surrogate training set and refit
    # (In reality, you'd accumulate all data across rounds)
    # Here, we just augment.
    # NOTE: in a more complete implementation you’d keep global X,y arrays.
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
):
    global GLOBAL_WT_SEQUENCE
    GLOBAL_WT_SEQUENCE = wt_seq

    print("Running initial round...")
    surrogate, archive, train_seqs, train_stab = initial_round(
        wt_seq,
        n_single=initial_n_single,
        n_double=initial_n_double,
        use_esm=use_esm,
        esm_model=esm_model,
        esm_batch_size=esm_batch_size,
        esm_device=esm_device,
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
            beta=max(0.5, 2.0 / (r + 1))  # anneal exploration
        )

        print("Selected batch for lab testing:")
        for s, stab in zip(result["batch_sequences"], result["batch_stabilities"]):
            muts = mutation_list(wt_seq, s)
            print(f" Sequence: {s}\n Mutations: {muts}\n Stability proxy: {stab:.3f}\n")

    # At the end you have:
    # - archive.elites(): diverse high-stability variants
    # - last selected batch: good lab candidates
    return archive


if __name__ == "__main__":
    # Example usage with a fake WT sequence (replace with real PETase)
    wt_example = "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
    # Set use_esm=True to encode variants with ESM embeddings instead of one-hot.
    # Requires `pip install fair-esm torch` beforehand.
    run_pipeline(
        wt_example,
        n_rounds=2,
        use_esm=True,
        esm_model="esm2_t6_8M_UR50D",
        esm_batch_size=8,
    )
