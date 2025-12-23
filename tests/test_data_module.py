# note: load sequences shoud be tested here as well
import pandas as pd

from src.data.preprocess_sequences import clean_sequences


def test_load_and_clean():
    df = pd.DataFrame({"id": ["A", "B"], "sequence": ["ACDE", "ACX!E"]})
    df_clean = clean_sequences(df)
    assert all(df_clean["sequence"].str.isalpha())
    assert df_clean.loc[1, "sequence"] == "ACE"
