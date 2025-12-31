"""Sanity checks for PETase mutation model behavior."""

from src.petase_pipeline import (
    FROZEN_POSITIONS,
    propose_double_mutants_guided,
    propose_new_candidates,
    propose_single_mutants_guided,
)
from src.petase_scoring import hamming_distance


def _wt_example() -> str:
    return (
        "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKW"
        "WGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFS"
        "SVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
    )


def _check_frozen_positions(wt_seq: str, variants: list[str]) -> None:
    for variant in variants:
        for frozen_pos in FROZEN_POSITIONS:
            idx = frozen_pos - 1
            if idx < len(variant) and variant[idx] != wt_seq[idx]:
                raise AssertionError(
                    f"Frozen position {frozen_pos} mutated: {wt_seq[idx]} -> {variant[idx]}"
                )


def main() -> None:
    wt_seq = _wt_example()

    singles = propose_single_mutants_guided(wt_seq)
    doubles = propose_double_mutants_guided(wt_seq, [], max_variants=50, max_dist=1.0)
    candidates = propose_new_candidates([wt_seq], n_candidates=50)

    print("wt_len", len(wt_seq))
    print("singles", len(singles), "min_dist", min(hamming_distance(wt_seq, s) for s in singles))
    print("doubles", len(doubles), "min_dist", min(hamming_distance(wt_seq, s) for s in doubles))
    print(
        "candidates",
        len(candidates),
        "min_dist",
        min(hamming_distance(wt_seq, s) for s in candidates),
    )

    all_variants = singles + doubles + candidates
    _check_frozen_positions(wt_seq, all_variants)

    print("frozen_positions_ok", True)


if __name__ == "__main__":
    main()
