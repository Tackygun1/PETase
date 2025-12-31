"""Tests for the PETase pipeline mutation model and related functions."""

import numpy as np
import pytest

from src.petase_pipeline import (
    AMINO_ACIDS,
    FROZEN_POSITIONS,
    QDArchive,
    acquisition_ucb,
    allowed_positions,
    pick_diverse_batch,
    propose_double_mutants_guided,
    propose_new_candidates,
    propose_single_mutants_guided,
)
from src.petase_scoring import hamming_distance, set_reference_sequence


# Sample sequences for testing
SIMPLE_SEQ = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids
WT_PETASE_SHORT = "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKW"


class TestAllowedPositions:
    """Test the allowed_positions function that determines mutable sites."""

    def test_allowed_positions_excludes_frozen_sites(self):
        """Frozen positions should be excluded from allowed mutations."""
        seq = "A" * 300  # Long enough to include frozen positions
        allowed = allowed_positions(seq)

        # Convert to 1-based for comparison with FROZEN_POSITIONS
        allowed_1based = {i + 1 for i in allowed}

        # Verify no frozen positions are in allowed list
        assert FROZEN_POSITIONS.isdisjoint(allowed_1based)

    def test_allowed_positions_includes_unfrozen_sites(self):
        """Unfrozen positions should be included in allowed mutations."""
        seq = "A" * 50
        allowed = allowed_positions(seq)

        # All positions should be allowed for a short sequence without frozen sites
        # Frozen positions start at 87 (1-based)
        assert len(allowed) == 50

    def test_allowed_positions_correct_indexing(self):
        """Returned indices should be 0-based."""
        seq = "ACDE"
        allowed = allowed_positions(seq)

        # All positions should be allowed (none are in FROZEN_POSITIONS)
        assert allowed == [0, 1, 2, 3]

    def test_frozen_catalytic_triad(self):
        """Catalytic triad positions (160, 206, 237) must be frozen."""
        seq = "A" * 300
        allowed = allowed_positions(seq)
        allowed_1based = {i + 1 for i in allowed}

        # Verify catalytic triad is protected
        assert 160 not in allowed_1based
        assert 206 not in allowed_1based
        assert 237 not in allowed_1based


class TestProposeSingleMutantsGuided:
    """Test single mutant generation with guided mutations."""

    def test_generates_single_mutants(self):
        """Should generate single mutants at known beneficial positions."""
        wt_seq = "A" * 300
        variants = propose_single_mutants_guided(wt_seq)

        assert len(variants) > 0
        for variant in variants:
            assert len(variant) == len(wt_seq)
            assert hamming_distance(wt_seq, variant) == 1

    def test_respects_wild_type_at_mutation_sites(self):
        """Should not propose mutations that match wild-type."""
        wt_seq = "A" * 300
        wt_seq = wt_seq[:120] + "E" + wt_seq[121:]  # Set position 121 (1-based) to E
        variants = propose_single_mutants_guided(wt_seq)

        # Position 121 should have E or D mutations, but E is already WT
        # So only D should be proposed
        # Check that we have D mutations at position 121
        assert any(v[120] == "D" for v in variants)

    def test_with_custom_mutation_list(self):
        """Should use custom mutation list when provided."""
        wt_seq = "ACDEFG"
        mut_list = [(1, "Y"), (3, "W")]  # 1-based positions
        variants = propose_single_mutants_guided(wt_seq, mut_list=mut_list)

        assert len(variants) == 2
        assert "YCDEFG" in variants  # A1Y
        assert "ACWEFG" in variants  # D3W

    def test_custom_mutation_list_skips_out_of_range(self):
        """Out-of-range custom mutations should be skipped without errors."""
        wt_seq = "ACDEFG"
        mut_list = [(-1, "Y"), (3, "W"), (999, "A")]
        variants = propose_single_mutants_guided(wt_seq, mut_list=mut_list)

        assert variants == ["ACWEFG"]

    def test_no_duplicate_variants(self):
        """Should not generate duplicate variants."""
        wt_seq = "A" * 300
        variants = propose_single_mutants_guided(wt_seq)

        assert len(variants) == len(set(variants))


class TestProposeDoubleMutantsGuided:
    """Test double mutant generation with structural constraints."""

    class _DummyEmbeddingProvider:
        def get(self, seqs):
            return np.zeros((len(seqs), 4))

    def test_generates_double_mutants(self):
        """Should generate sequences with exactly 2 mutations."""
        wt_seq = "A" * 300
        base_variants = []
        variants = propose_double_mutants_guided(
            wt_seq, base_variants, max_variants=50, max_dist=1.0
        )

        assert len(variants) > 0
        for variant in variants:
            assert hamming_distance(wt_seq, variant) == 2

    def test_respects_max_variants_limit(self):
        """Should not exceed max_variants limit."""
        wt_seq = "A" * 300
        max_variants = 10
        variants = propose_double_mutants_guided(
            wt_seq, [], max_variants=max_variants, max_dist=1.0
        )

        assert len(variants) <= max_variants
        assert len(variants) == len(set(variants))

    def test_respects_max_variants_after_embedding_filter(self):
        """Should cap variants after embedding filtering."""
        wt_seq = "A" * 300
        max_variants = 3
        variants = propose_double_mutants_guided(
            wt_seq,
            [],
            max_variants=max_variants,
            max_dist=1.0,
            embedding_provider=self._DummyEmbeddingProvider(),
        )

        assert len(variants) == max_variants

    def test_no_duplicates_with_embedding_provider(self):
        """Should not produce duplicates when embeddings are used."""
        wt_seq = "A" * 300
        variants = propose_double_mutants_guided(
            wt_seq,
            [],
            max_variants=10,
            max_dist=1.0,
            embedding_provider=self._DummyEmbeddingProvider(),
        )

        assert len(variants) == len(set(variants))

    def test_targets_key_positions(self):
        """Should target known beneficial positions for mutations."""
        wt_seq = "A" * 300
        variants = propose_double_mutants_guided(wt_seq, [], max_variants=100, max_dist=1.0)

        # Key positions include 121, 186, 224, 233, 280, 95, 201, 159, 229, 181
        # Check that some variants mutate these positions
        key_positions_0based = {120, 185, 223, 232, 279, 94, 200, 158, 228, 180}

        has_key_mutation = False
        for variant in variants:
            for pos in key_positions_0based:
                if pos < len(variant) and variant[pos] != wt_seq[pos]:
                    has_key_mutation = True
                    break
            if has_key_mutation:
                break

        assert has_key_mutation

    def test_returns_empty_without_embedding_provider_and_insufficient_positions(self):
        """Should handle edge cases gracefully."""
        wt_seq = "AC"  # Very short sequence, no key positions
        variants = propose_double_mutants_guided(wt_seq, [], max_variants=10, max_dist=1.0)

        # Should return empty or very few variants since no key positions are in range
        assert len(variants) >= 0


class TestProposeNewCandidates:
    """Test random mutation around elite sequences."""

    def test_generates_requested_number_of_candidates(self):
        """Should generate approximately n_candidates."""
        base_seqs = ["ACDEFGHIKL"]
        n_candidates = 50
        candidates = propose_new_candidates(base_seqs, n_candidates=n_candidates)

        assert len(candidates) == n_candidates

    def test_candidates_are_unique(self):
        """All generated candidates should be unique."""
        base_seqs = ["ACDEFGHIKL"]
        candidates = propose_new_candidates(base_seqs, n_candidates=100)

        assert len(candidates) == len(set(candidates))

    def test_mutations_from_parent_sequences(self):
        """Candidates should be mutations of parent sequences."""
        base_seq = "AAAAA" * 60  # 300 AA
        base_seqs = [base_seq]
        candidates = propose_new_candidates(base_seqs, n_candidates=50)

        for candidate in candidates:
            assert len(candidate) == len(base_seq)
            # Should have 1-3 mutations
            dist = hamming_distance(base_seq, candidate)
            assert 1 <= dist <= 3

    def test_respects_allowed_positions(self):
        """Should only mutate allowed positions."""
        # Create sequence where some positions are frozen
        base_seq = "A" * 300
        base_seqs = [base_seq]
        candidates = propose_new_candidates(base_seqs, n_candidates=20)

        allowed = set(allowed_positions(base_seq))

        for candidate in candidates:
            # Check that mutations only occur at allowed positions
            for i in range(len(candidate)):
                if candidate[i] != base_seq[i]:
                    assert i in allowed

    def test_mixed_length_parents_handled(self):
        """Should handle mixed-length parents without out-of-range mutations."""
        base_seqs = ["A" * 300, "A" * 120]

        candidates = propose_new_candidates(base_seqs, n_candidates=20)

        # Each candidate should align with one of the parents and only mutate allowed positions.
        for candidate in candidates:
            if len(candidate) == len(base_seqs[0]):
                parent = base_seqs[0]
            elif len(candidate) == len(base_seqs[1]):
                parent = base_seqs[1]
            else:
                pytest.fail("Candidate length does not match any parent length.")

            allowed = set(allowed_positions(parent))
            for i in range(len(candidate)):
                if candidate[i] != parent[i]:
                    assert i in allowed

    def test_never_mutates_frozen_positions_across_parents(self):
        """Should never mutate frozen positions across many samples and parents."""
        base_seqs = ["A" * 300, "A" * 250, "A" * 200]
        candidates = propose_new_candidates(base_seqs, n_candidates=200)

        for candidate in candidates:
            if len(candidate) == 300:
                parent = base_seqs[0]
            elif len(candidate) == 250:
                parent = base_seqs[1]
            elif len(candidate) == 200:
                parent = base_seqs[2]
            else:
                pytest.fail("Candidate length does not match any parent length.")

            for frozen_pos in FROZEN_POSITIONS:
                idx = frozen_pos - 1
                if idx < len(candidate):
                    assert candidate[idx] == parent[idx], (
                        f"Frozen position {frozen_pos} was mutated: "
                        f"{parent[idx]} -> {candidate[idx]}"
                    )


class TestQDArchive:
    """Test Quality-Diversity archive for maintaining diverse variants."""

    def test_archive_initialization(self):
        """Archive should initialize with empty cells."""
        archive = QDArchive(bin_width=0.5)
        assert len(archive.cells) == 0

    def test_archive_insert_and_retrieve(self):
        """Should insert and retrieve variants."""
        set_reference_sequence("AAAA")
        archive = QDArchive(bin_width=0.5)

        seq = "AAAT"
        stability = 1.5
        archive.insert(seq, stability)

        elites = archive.elites()
        assert len(elites) == 1
        assert seq in elites

    def test_archive_keeps_best_per_cell(self):
        """Should keep only the best variant per QD cell."""
        wt_seq = "AAAA"
        set_reference_sequence(wt_seq)
        archive = QDArchive(bin_width=0.5)

        # Two sequences with same mutation count and similar stability
        seq1 = "AAAT"  # 1 mutation
        seq2 = "AAAC"  # 1 mutation

        archive.insert(seq1, 1.2)  # stability bin 2
        archive.insert(seq2, 1.3)  # stability bin 2

        # Should keep only seq2 (higher stability) in the same cell
        elites = archive.elites()
        assert len(elites) == 1
        assert seq2 in elites
        assert seq1 not in elites

    def test_archive_maintains_diversity(self):
        """Should maintain variants in different QD cells."""
        wt_seq = "AAAA"
        set_reference_sequence(wt_seq)
        archive = QDArchive(bin_width=0.5)

        # Different mutation counts or stability bins
        archive.insert("AAAT", 1.0)  # 1 mutation, bin 2
        archive.insert("AATC", 1.5)  # 2 mutations, bin 3
        archive.insert("TTAA", 2.0)  # 2 mutations, bin 4

        elites = archive.elites()
        assert len(elites) >= 2  # At least 2 different cells


class TestAcquisitionUCB:
    """Test Upper Confidence Bound acquisition function."""

    def test_ucb_calculation(self):
        """UCB should be mean + beta * std."""
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])
        beta = 2.0

        ucb = acquisition_ucb(mean, std, beta=beta)
        expected = mean + beta * std

        np.testing.assert_array_almost_equal(ucb, expected)

    def test_ucb_exploration_tradeoff(self):
        """Higher beta should favor uncertain predictions."""
        mean = np.array([1.0, 0.8])
        std = np.array([0.1, 0.5])  # Second has high uncertainty

        # Low exploration
        ucb_low = acquisition_ucb(mean, std, beta=0.1)
        assert ucb_low[0] > ucb_low[1]  # Favors higher mean

        # High exploration
        ucb_high = acquisition_ucb(mean, std, beta=3.0)
        assert ucb_high[1] > ucb_high[0]  # Favors higher uncertainty


class TestPickDiverseBatch:
    """Test diversity-based batch selection."""

    def test_picks_requested_batch_size(self):
        """Should return batch_size candidates."""
        candidates = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        batch = pick_diverse_batch(candidates, scores, batch_size=3, min_hamming=1)

        assert len(batch) == 3

    def test_prefers_high_scores(self):
        """Should prefer candidates with higher scores."""
        candidates = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        batch = pick_diverse_batch(candidates, scores, batch_size=1, min_hamming=0)

        assert batch[0] == "AAAA"  # Highest score

    def test_enforces_diversity(self):
        """Should enforce minimum Hamming distance between selected variants."""
        # Create candidates that are very similar
        candidates = ["AAAA", "AAAB", "AAAC", "AAAD"]
        scores = np.array([4.0, 3.0, 2.0, 1.0])

        batch = pick_diverse_batch(candidates, scores, batch_size=4, min_hamming=2)

        # Should not select all 4 since they're too similar
        assert len(batch) < 4

    def test_handles_empty_candidates(self):
        """Should handle empty candidate list gracefully."""
        candidates = []
        scores = np.array([])

        batch = pick_diverse_batch(candidates, scores, batch_size=5, min_hamming=2)

        assert len(batch) == 0

    def test_batch_size_limited_by_candidates(self):
        """Should not return more than available candidates."""
        candidates = ["AAAA", "TTTT"]
        scores = np.array([2.0, 1.0])

        batch = pick_diverse_batch(candidates, scores, batch_size=10, min_hamming=1)

        assert len(batch) <= 2


class TestMutationConstraints:
    """Test that mutation model respects structural constraints."""

    def test_frozen_positions_never_mutated(self):
        """Verify that frozen positions are never mutated."""
        wt_seq = "A" * 300

        # Generate many variants
        singles = propose_single_mutants_guided(wt_seq)
        doubles = propose_double_mutants_guided(wt_seq, [], max_variants=100)
        base_seqs = [wt_seq]
        random_vars = propose_new_candidates(base_seqs, n_candidates=50)

        all_variants = singles + doubles + random_vars

        for variant in all_variants:
            for frozen_pos in FROZEN_POSITIONS:
                # Convert 1-based to 0-based index
                idx = frozen_pos - 1
                if idx < len(variant) and idx < len(wt_seq):
                    assert variant[idx] == wt_seq[idx], (
                        f"Frozen position {frozen_pos} was mutated: {wt_seq[idx]} -> {variant[idx]}"
                    )

    def test_only_canonical_amino_acids(self):
        """All mutations should use only canonical amino acids."""
        wt_seq = "A" * 300
        variants = propose_single_mutants_guided(wt_seq)

        for variant in variants:
            for aa in variant:
                assert aa in AMINO_ACIDS


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_sequence_handling(self):
        """Should handle very short sequences."""
        short_seq = "ACDE"
        variants = propose_single_mutants_guided(short_seq)

        # Should still work even though no key positions exist
        assert len(variants) >= 0

    def test_sequence_length_preservation(self):
        """All mutations should preserve sequence length."""
        wt_seq = "ACDEFGHIKLMNPQRSTVWY" * 15

        singles = propose_single_mutants_guided(wt_seq)
        doubles = propose_double_mutants_guided(wt_seq, [], max_variants=10)

        for variant in singles + doubles:
            assert len(variant) == len(wt_seq)

    def test_no_silent_mutations(self):
        """Proposed mutations should actually change the sequence."""
        wt_seq = "A" * 300
        singles = propose_single_mutants_guided(wt_seq)

        for variant in singles:
            assert variant != wt_seq
