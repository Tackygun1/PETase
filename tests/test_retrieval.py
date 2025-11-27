import numpy as np

from acquisition.retrieval import cosine_search, load_ref_embeddings, normalize, radius_filter


def test_cosine_search_and_radius(tmp_path):
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    npz_path = tmp_path / "ref.npz"
    np.savez(npz_path, **emb)

    mat, ids = load_ref_embeddings(npz_path)
    mat = normalize(mat)
    query = np.array([0.9, 0.1])
    neighbors = cosine_search(query, mat, ids, top_k=2)
    assert neighbors[0][0] == "a"
    filtered = radius_filter(neighbors, min_sim=0.5)
    assert filtered and filtered[0][0] == "a"
