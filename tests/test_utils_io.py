from pathlib import Path

import numpy as np
from utils.io import load_embeddings, load_labels_csv


def test_load_labels_csv_success(tmp_path: Path):
    data = "id,label\nseq1,0.5\nseq2,1.2\n"
    p = tmp_path / "labels.csv"
    p.write_text(data)

    df = load_labels_csv(p)
    assert list(df.columns) == ["id", "label"]
    assert df.shape[0] == 2
    assert df.loc[0, "id"] == "seq1"


def test_load_labels_csv_missing_columns(tmp_path: Path):
    # missing the 'label' column
    data = "id,value\nseq1,0.5\n"
    p = tmp_path / "bad_labels.csv"
    p.write_text(data)

    try:
        load_labels_csv(p)
        raised = False
    except ValueError as e:
        raised = True
        assert "missing columns" in str(e)

    assert raised


def test_load_embeddings_npz(tmp_path: Path):
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    p = tmp_path / "embs.npz"
    np.savez(p, seq1=a, seq2=b)

    embs = load_embeddings(p)
    assert set(embs.keys()) == {"seq1", "seq2"}
    assert np.allclose(embs["seq1"], a)
