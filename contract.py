import os
import numpy as np
import pandas as pd

EMBEDDING_DIM = 32
MAX_SEQ_LEN = 50

META_COLUMNS = [
    "sequence_id",
    "sender_id",
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


def generate_mock_stage_a_output(n_sequences: int = 300, out_dir: str = "stage_a_output_mock", seed: int = 42):
    rng = np.random.default_rng(seed)

    seq_lens = rng.integers(3, MAX_SEQ_LEN + 1, size=n_sequences)
    splits = rng.choice(["train", "val", "test"], size=n_sequences, p=[0.7, 0.15, 0.15])
    labels = rng.choice([0.0, 1.0, np.nan], size=n_sequences, p=[0.85, 0.05, 0.10])
    recon_error = rng.exponential(scale=0.05, size=n_sequences)
    anomaly_score = (recon_error - recon_error.min()) / (np.ptp(recon_error) + 1e-9)

    meta_df = pd.DataFrame({
        "sequence_id": [f"mockSender{i}_0" for i in range(n_sequences)],
        "sender_id": [f"mockSender{i}" for i in range(n_sequences)],
        "split": splits,
        "seq_len": seq_lens,
        "label": labels,
        "recon_error": recon_error,
        "anomaly_score": anomaly_score,
    })

    embeddings = rng.normal(size=(n_sequences, EMBEDDING_DIM)).astype("float32")

    per_step_error = np.full((n_sequences, MAX_SEQ_LEN), np.nan, dtype="float32")
    for i, L in enumerate(seq_lens):
        per_step_error[i, :L] = rng.exponential(scale=0.05, size=L)

    save_stage_a_output(meta_df, embeddings, per_step_error, out_dir=out_dir)
    return meta_df, embeddings, per_step_error


if __name__ == "__main__":
    meta_df, embeddings, per_step_error = generate_mock_stage_a_output()
    meta_df2, embeddings2, per_step_error2 = load_stage_a_output("stage_a_output_mock")
    pd.testing.assert_frame_equal(meta_df, meta_df2, check_exact=False)
    assert np.array_equal(embeddings, embeddings2)
    assert np.allclose(per_step_error, per_step_error2, equal_nan=True)
