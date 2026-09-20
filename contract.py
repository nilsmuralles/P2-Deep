import os
import numpy as np
import pandas as pd

EMBEDDING_DIM = 32
MAX_SEQ_LEN = 50

META_COLUMNS = [
    "sequence_id",
    "account_id",
    "split",
    "seq_len",
    "label",
    "recon_error",
    "anomaly_score",
]

def save_stage_a_output(meta_df: pd.DataFrame, embeddings: np.ndarray, per_step_error: np.ndarray, out_dir: str = "stage_a_output"):
    assert list(meta_df.columns) == META_COLUMNS, f"Columnas deben ser exactamente {META_COLUMNS}"
    n = len(meta_df)
    assert embeddings.shape == (n, EMBEDDING_DIM), f"embeddings debe ser ({n}, {EMBEDDING_DIM})"
    assert per_step_error.shape == (n, MAX_SEQ_LEN), f"per_step_error debe ser ({n}, {MAX_SEQ_LEN})"

    os.makedirs(out_dir, exist_ok=True)
    meta_df.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings.astype("float32"))
    np.save(os.path.join(out_dir, "per_step_error.npy"), per_step_error.astype("float32"))
    print(f"Guardado en {out_dir}/: meta.csv, embeddings.npy, per_step_error.npy ({n} secuencias)")


def load_stage_a_output(in_dir: str = "stage_a_output"):
    meta_df = pd.read_csv(os.path.join(in_dir, "meta.csv"))
    embeddings = np.load(os.path.join(in_dir, "embeddings.npy"))
    per_step_error = np.load(os.path.join(in_dir, "per_step_error.npy"))
    return meta_df, embeddings, per_step_error


