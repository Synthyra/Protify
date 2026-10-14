import os
import time
import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from .data_loader import load_proteingym_dms
from .dms_ids import ALL_SUBSTITUTION_DMS_IDS
from .scorer import ProteinGymScorer
from ...base_models.get_base_models import get_base_model


def compare_scoring_methods(
    model_names: list[str],
    device: str | None = None,
    methods: list[str] | None = None,
    dms_ids: list[str] | None = None,
    progress: bool = True,
    output_csv: str | None = None,
    batch_size: int = 32,
    scoring_window: str = "optimal"
) -> pd.DataFrame:
    """
    Compare scoring methods across one or more models and DMS assays.
    
    Args:
        model_names: List of model names to evaluate
        device: Device string like 'cuda' or 'cpu'
        methods: List of scoring methods to compare
        dms_ids: List of DMS IDs to evaluate
        progress: Whether to show progress bars
        output_csv: Optional path to save results CSV
        batch_size: Batch size for inference (default: 32)
        scoring_window: Windowing strategy ('optimal' or 'sliding')
        
    Returns:
        DataFrame with model_name, scoring_method, Average_Spearman, Average_Time_Seconds, Total_Time_Seconds, and n_assays columns
    """
    if methods is None:
        methods = ["masked_marginal", "mutant_marginal", "wildtype_marginal", "global_log_prob"]
    
    if dms_ids is None:
        dms_ids = ALL_SUBSTITUTION_DMS_IDS
    
    all_summary_results = []
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Store results for each assay
            assay_results = []
            spearman_results = []
            timing_results = []
            # Load model once per model and reuse across methods/assays
            model, tokenizer = get_base_model(model_name, masked_lm=True)
            model = model.to(device).eval()
            scorer = ProteinGymScorer(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
            )
            
            for dms_id in dms_ids:
                print(f"\nProcessing DMS ID: {dms_id}")
                df = load_proteingym_dms(
                    dms_id=dms_id,
                    mode="benchmark",
                    repo_id="GleghornLab/ProteinGym_DMS",
                )
                    
                assay_result = df.copy()
                assay_result['dms_id'] = dms_id
                
                for method in methods:
                    print(f"Running {method} scoring for {dms_id}...")
                    
                    # Measure timing for this scoring method
                    start_time = time.time()
                    scored_df = scorer.score_substitutions(
                        df=df,
                        scoring_method=method,
                        scoring_window=scoring_window,
                    )
                    end_time = time.time()
                    method_duration = end_time - start_time
                    
                    print(f"  {method} scoring completed in {method_duration:.2f} seconds")
                    
                    assay_result[f'{method}_score'] = scored_df['delta_log_prob']
                    
                    # Calculate Spearman
                    x = scored_df["delta_log_prob"].to_numpy()
                    y = scored_df["DMS_score"].to_numpy()
                    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                        print(f"No valid scores for {method} scoring for {dms_id}")
                        spearman_rho = np.nan
                    else:
                        mask = ~(np.isnan(x) | np.isnan(y))
                        if mask.sum() < 2:
                            print(f"Not enough valid scores for {method} scoring for {dms_id}")
                            spearman_rho = np.nan
                        else:
                            rho, _ = spearmanr(x[mask], y[mask])
                            spearman_rho = rho
                            print(f"Spearman correlation for {method} on {dms_id}: {rho:.4f}")
                    
                    assay_result[f'{method}_spearman_rho'] = spearman_rho
                    
                    # Store for summary calculation
                    spearman_results.append({
                        'dms_id': dms_id,
                        'method': method,
                        'spearman_rho': spearman_rho
                    })
                    
                    # Store timing results
                    timing_results.append({
                        'dms_id': dms_id,
                        'method': method,
                        'duration_seconds': method_duration
                    })
                        
                assay_results.append(assay_result)
            
            # Calculate average Spearman correlations and timing for this model
            spearman_df = pd.DataFrame(spearman_results)
            timing_df = pd.DataFrame(timing_results)
            summary_results = []
            
            for method in methods:
                method_data = spearman_df[spearman_df['method'] == method]['spearman_rho']
                valid_correlations = method_data[~np.isnan(method_data)]
                
                if len(valid_correlations) > 0:
                    avg_spearman = valid_correlations.mean()
                    n_assays = len(valid_correlations)
                else:
                    avg_spearman = np.nan
                    n_assays = 0
                
                # Calculate timing statistics for this method
                method_timing_data = timing_df[timing_df['method'] == method]['duration_seconds']
                if len(method_timing_data) > 0:
                    avg_time = method_timing_data.mean()
                    total_time = method_timing_data.sum()
                else:
                    avg_time = np.nan
                    total_time = np.nan
                
                summary_results.append({
                    'model_name': model_name,
                    'scoring_method': method,
                    'Average_Spearman': avg_spearman,
                    'Average_Time_Seconds': avg_time,
                    'Total_Time_Seconds': total_time,
                    'n_assays': n_assays
                })
            
            model_summary_df = pd.DataFrame(summary_results)
            all_summary_results.append(model_summary_df)
            
            # Print summary for this model
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR MODEL: {model_name}")
            print(f"{'='*60}")
            print(model_summary_df.to_string(index=False))
            
        except Exception as e:  # noqa: broad-except a failed model reports and gets an empty summary so the comparison continues
            print(f"Error processing model {model_name}: {e}")
            # Create empty summary for failed model
            failed_summary = pd.DataFrame([{
                'model_name': model_name,
                'scoring_method': method,
                'Average_Spearman': np.nan,
                'Average_Time_Seconds': np.nan,
                'Total_Time_Seconds': np.nan,
                'n_assays': 0
            } for method in methods])
            all_summary_results.append(failed_summary)
    
    # Combine all summary results
    if all_summary_results:
        combined_summary = pd.concat(all_summary_results, ignore_index=True)
    else:
        combined_summary = pd.DataFrame()
    
    # Save results if output path provided
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        combined_summary.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - AVERAGE SPEARMAN CORRELATIONS")
    print(f"{'='*80}")
    print(combined_summary.to_string(index=False))
    
    return combined_summary
