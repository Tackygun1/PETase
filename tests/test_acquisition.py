import numpy as np

from src.acquisition.acquisition import Candidate, apply_mutations, compute_acquisition, filter_by_distance
from src.acquisition.qd_archive import QDArchive
from src.acquisition.proposer import propose_from_neighbors


def test_apply_mutations_single():
    seq = "ACDE"
    mutated = apply_mutations(seq, ["C2Y"])
    assert mutated == "AYDE"


def test_apply_mutations_bad_origin_raises():
    seq = "ACDE"
    try:
        apply_mutations(seq, ["A2Y"])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_compute_acquisition_with_activity_floor():
    c1 = Candidate(
        seq_id="s1",
        sequence="AAAA",
        mutations=[],
        mut_count=0,
        pred_stab_mean=1.0,
        pred_stab_std=0.5,
        pred_act_mean=0.5,
    )
    c2 = Candidate(
        seq_id="s2",
        sequence="AAAT",
        mutations=["A4T"],
        mut_count=1,
        pred_stab_mean=0.8,
        pred_stab_std=0.1,
        pred_act_mean=0.1,
    )
    scored = compute_acquisition([c1, c2], beta=1.0, w_stability=1.0, w_activity=1.0, activity_floor=0.2)
    acq1 = scored[0].acquisition
    acq2 = scored[1].acquisition
    assert acq1 > 0
    assert acq2 == -np.inf


def test_qd_archive_and_distance_filter():
    archive = QDArchive(max_mutations=2, stability_bin_width=0.5)
    c1 = Candidate(seq_id="a", sequence="AAAA", mutations=[], mut_count=0, pred_stab_mean=0.9, pred_stab_std=0.1, acquisition=1.0)
    c2 = Candidate(seq_id="b", sequence="AAAT", mutations=["A4T"], mut_count=1, pred_stab_mean=0.4, pred_stab_std=0.1, acquisition=0.5)
    archive.maybe_insert(c1.mut_count, c1.pred_stab_mean, c1.acquisition, {"seq_id": c1.seq_id})
    archive.maybe_insert(c2.mut_count, c2.pred_stab_mean, c2.acquisition, {"seq_id": c2.seq_id})
    elites = archive.elites()
    assert len(elites) == 2
    filtered = filter_by_distance([c1, c2], min_hamming=2)
    assert len(filtered) == 1  # hamming distance is 1, so second candidate is dropped


def test_propose_from_neighbors_respects_protected_sites():
    parent_seq = "ACDE"
    neighbors = ["ACDF"]  # mutation at position 4 allowed
    neighbor_ids = ["n1"]
    cands = propose_from_neighbors("p", parent_seq, neighbors, neighbor_ids, max_mutations=1)
    assert len(cands) == 1
    assert cands[0].mutations == ["E4F"]
