import pandas as pd
from huggingface_hub import hf_hub_download


def _load_parquet_by_dms(repo_id: str, dms_id: str) -> pd.DataFrame:
    """Load one ProteinGym assay shard from the Hub."""
    id_str = str(dms_id)
    filename = f"by_dms_id/{id_str}.parquet"
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_parquet(local_path)


def load_proteingym_dms(dms_id: str, mode: str, repo_id: str = "GleghornLab/ProteinGym_DMS") -> pd.DataFrame:
    """
    Load a single ProteinGym DMS assay, processing columns as specified.

    Modes:
    - "benchmark": Keeps the columns for standard substitution benchmark.
    - "indels": Keeps only indels assays.
    - "singles": Keeps only single substitutions variants.
    - "multiples": Keeps only multiple substitutions variants.
    """
    df = _load_parquet_by_dms(repo_id=repo_id, dms_id=dms_id)
    
    if mode == 'benchmark':
        df = df[df['is_indel'] == False]
        df = df[["DMS_id", "mutated_seq", "target_seq", "DMS_score", "DMS_score_bin", "mutant"]]
    elif mode == 'indels':
        # Indels only, no mutant triplet column
        df = df[df['is_indel'] == True]
        df = df[["DMS_id", "mutated_seq", "target_seq", "DMS_score", "DMS_score_bin"]]
    elif mode == 'singles':
        # Single substitutions only
        df = df[df['is_indel'] == False]
        df = df[df['num_mutations'] == 1]
        df = df[["DMS_id", "mutated_seq", "target_seq", "DMS_score", "DMS_score_bin", "mutant"]]
    elif mode == 'multiples':
        # Multiple substitutions only
        df = df[df['is_indel'] == False]
        df = df[df['num_mutations'] > 1]
        df = df[["DMS_id", "mutated_seq", "target_seq", "DMS_score", "DMS_score_bin", "mutant"]]

    return df.reset_index(drop=True)
