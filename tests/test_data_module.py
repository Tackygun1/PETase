# note: load sequences shoud be tested here as well
import pandas as pd


def test_load_and_clean():
    """Placeholder test - data module not yet implemented."""
    df = pd.DataFrame({"id": ["A", "B"], "sequence": ["ACDE", "ACDE"]})
    # TODO: Implement clean_sequences function in utils module
    assert all(df["sequence"].str.isalpha())
