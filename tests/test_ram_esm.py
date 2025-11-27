import numpy as np

from acquisition.ram_esm import build_ram_scorer


def test_ram_scorer_soft_topk(tmp_path):
    # Reference bank: two sequences with different stability labels
    ref_embs = {
        "ref1": np.array([1.0, 0.0], dtype=float),
        "ref2": np.array([0.0, 1.0], dtype=float),
    }
    ref_npz = tmp_path / "ref.npz"
    np.savez(ref_npz, **ref_embs)

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("id,stability\nref1,1.0\nref2,0.0\n")

    scorer = build_ram_scorer(ref_npz, labels_csv, top_k=1, temperature=0.05)

    class Cand:
        def __init__(self, seq_id):
            self.seq_id = seq_id

    # Candidate closer to ref1 should score near 1.0
    cand = Cand("cand1")
    cand_embs = {"cand1": np.array([0.9, 0.1], dtype=float)}
    scores = scorer([cand], cand_embs)
    assert scores.shape == (1,)
    assert scores[0] > 0.8
