"""Batch design logic combining acquisition scores with quality-diversity."""

from __future__ import annotations

from typing import List, Optional

from .acquisition import Candidate, compute_acquisition, filter_by_distance
from .qd_archive import QDArchive


def design_batch(
    candidates: List[Candidate],
    archive: QDArchive,
    batch_size: int,
    beta: float = 1.0,
    w_stability: float = 1.0,
    w_activity: float = 0.0,
    w_ram: float = 0.0,
    activity_floor: Optional[float] = None,
    min_hamming: int = 2,
) -> List[Candidate]:
    """Score candidates, update the QD archive, and return a diverse batch."""
    scored = compute_acquisition(
        candidates,
        beta=beta,
        w_stability=w_stability,
        w_activity=w_activity,
        w_ram=w_ram,
        activity_floor=activity_floor,
    )

    # Update archive and capture elites by niche.
    for cand in scored:
        if cand.acquisition is None or cand.acquisition == -float("inf"):
            continue
        archive.maybe_insert(
            mutation_count=cand.mut_count,
            stability_score=cand.pred_stab_mean,
            score=cand.acquisition,
            payload={
                "seq_id": cand.seq_id,
                "mutations": cand.mutations,
                "acquisition": cand.acquisition,
                "pred_stab_mean": cand.pred_stab_mean,
                "pred_stab_std": cand.pred_stab_std,
                "pred_act_mean": cand.pred_act_mean,
                "ram_score": cand.ram_score,
            },
        )

    elites = archive.elites()
    ordered = [e.payload for e in elites]

    # Map payloads back to Candidate objects where possible.
    elite_candidates = []
    payload_lookup = {(c.seq_id, tuple(c.mutations)): c for c in candidates}
    for p in ordered:
        key = (p["seq_id"], tuple(p["mutations"]))
        if key in payload_lookup:
            elite_candidates.append(payload_lookup[key])

    diverse = filter_by_distance(elite_candidates, min_hamming=min_hamming)
    return diverse[:batch_size]
