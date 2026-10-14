'''
Adapted from https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_DMS_benchmarks.py
Changes:
- Added ability to restrict analysis to a subset of DMS IDs, when they are specified in main.py
- Evaluates only models that are specified in main.py
'''

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd

from typing import Any
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef


warnings.simplefilter(action='ignore', category=FutureWarning)


def minmax(x: np.ndarray) -> np.ndarray:
    # x: (n,)
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) )  # (n,)


def calc_ndcg(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series, **kwargs: Any) -> float:
    # y_true: (n,); y_score: (n,)
    '''
    Inputs:
        y_true: an array of the true scores where higher score is better
        y_score: an array of the predicted scores where higher score is better
    Options:
        quantile: If True, uses the top k quantile of the distribution
        top: under the quantile setting this is the top quantile to
            keep in the gains calc. This is a PERCENTAGE (i.e input 10 for top 10%)
    Notes:
        Currently we're calculating NDCG on the continuous value of the DMS
        I tried it on the binary value as well and the metrics seemed mostly
        the same.
    '''
    if 'quantile' not in kwargs:
        kwargs['quantile'] = True
    if 'top' not in kwargs:
        kwargs['top'] = 10
    if kwargs['quantile']:
        k = np.floor(y_true.shape[0]*(kwargs['top']/100)).astype(int)
    else:
        k = kwargs['top']
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    gains = minmax(y_true)  # (n,)
    ranks = np.argsort(np.argsort(-y_score)) + 1  # (n,)

    if k == 'all':
        k = len(ranks)
    #sub to top k
    ranks_k = ranks[ranks <= k]  # (k',)
    gains_k = gains[ranks <= k]  # (k',)
    #all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]  # (k'',)
    gains_fil = gains_k[gains_k != 0]  # (k'',)

    #if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return (0)
    
    #discounted cumulative gains
    dcg = np.sum([g/np.log2(r+1) for r,g in zip(ranks_fil, gains_fil, strict=True)])
    
    #ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1  # (n,)
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]  # (k',)
    ideal_gains_k = gains[ideal_ranks <= k]  # (k',)
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]  # (k'',)
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]  # (k'',)
    idcg = np.sum([g/np.log2(r+1) for r,g in zip(ideal_ranks_fil, ideal_gains_fil, strict=True)])

    #normalize
    ndcg = dcg/idcg
    
    return (ndcg)


def calc_toprecall(
    true_scores: np.ndarray | pd.Series,
    model_scores: np.ndarray | pd.Series,
    top_true: float = 10,
    top_model: float = 10,
) -> float:
    # true_scores: (n,); model_scores: (n,); top_true and top_model are percentages
    top_true = (true_scores >= np.percentile(true_scores, 100-top_true))  # (n,), bool
    top_model = (model_scores >= np.percentile(model_scores, 100-top_model))  # (n,), bool

    TP = (top_true) & (top_model)  # (n,), bool
    recall = TP.sum() / (top_true.sum()) if top_true.sum() > 0 else 0
    
    return (recall)


def standardization(x: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Assumes input is numpy array or pandas series"""
    # x: (n,)
    return (x - x.mean()) / x.std()  # (n,)


def compute_bootstrap_standard_error(df: pd.DataFrame, number_assay_reshuffle: int = 10000) -> pd.Series:
    """
    Computes the non-parametric bootstrap standard error for the mean estimate of a given performance metric (eg., Spearman, AUC) across DMS assays (ie., the sample standard deviation of the mean across bootstrap samples)
    """
    model_names = df.columns
    mean_performance_across_samples = []
    for _ in range(number_assay_reshuffle):
        mean_performance_across_samples.append(df.sample(frac=1.0, replace=True).mean(axis=0)) #Resample a dataset of the same size (with replacement) then take the sample mean
    mean_performance_across_samples=pd.DataFrame(data=mean_performance_across_samples,columns=model_names)
    return mean_performance_across_samples.std(ddof=1)


def compute_bootstrap_standard_error_functional_categories(
    df: pd.DataFrame, number_assay_reshuffle: int = 10000
) -> pd.Series:
    """
    Computes the non-parametric bootstrap standard error for the mean estimate of a given performance metric (eg., Spearman, AUC) across DMS assays (ie., the sample standard deviation of the mean across bootstrap samples)
    """
    mean_performance_across_samples = {}
    for category, group in df.groupby("Selection Type"):
        mean_performance_across_samples[category] = []
        for _ in range(number_assay_reshuffle):
            mean_performance_across_samples[category].append(group.sample(frac=1.0, replace=True).mean(axis=0)) #Resample a dataset of the same size (with replacement) then take the sample mean
        mean_performance_across_samples[category]=pd.DataFrame(data=mean_performance_across_samples[category])
    categories = list(mean_performance_across_samples.keys())
    combined_averages = mean_performance_across_samples[categories[0]].copy()
    for category in categories[1:]:
        combined_averages += mean_performance_across_samples[category]
    combined_averages /= len(categories)
    return combined_averages.std(ddof=1)


proteingym_folder_path = os.path.dirname(os.path.realpath(__file__))

def main() -> None:
    parser = argparse.ArgumentParser(description='ProteinGym performance analysis')
    parser.add_argument('--input_scoring_files_folder', type=str, help='Name of folder where all input scores are present (expects one scoring file per DMS)')
    parser.add_argument('--output_performance_file_folder', default='./outputs/tranception_performance', type=str, help='Name of folder where to save performance analysis files')
    parser.add_argument('--DMS_reference_file_path', type=str, help='Reference file with list of DMSs to consider')
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    parser.add_argument('--performance_by_depth', action='store_true', help='Whether to compute performance by mutation depth')
    parser.add_argument('--config_file', default=f'{os.path.dirname(proteingym_folder_path)}/config.json', type=str, help='Path to config file containing model information')
    parser.add_argument('--selected_model_names', nargs='+', default=None, help='Required to obtain column names from config file')
    parser.add_argument('--dms_ids', nargs='+', default=None, help='Subset of DMS ids to include; if omitted, all DMS in the reference are used')
    parser.add_argument('--scoring_method', choices=['masked_marginal','wildtype_marginal','mutant_marginal','pll','global_log_prob'], default='masked_marginal', help='Which zero-shot scoring file suffix to read')
    args = parser.parse_args()
    
    dms_reference = pd.read_csv(args.DMS_reference_file_path)
    # Optionally restrict analysis to a subset of DMS IDs
    if args.dms_ids is not None and len(args.dms_ids) > 0:
        requested_dms_ids = set(str(x) for x in args.dms_ids)
        dms_reference = dms_reference[dms_reference['DMS_id'].astype(str).isin(requested_dms_ids)]
        if dms_reference.empty:
            print("No matching DMS ids after filtering; nothing to compute.")
            return
    dms_reference["MSA_Neff_L_category"] = dms_reference["MSA_Neff_L_category"].apply(lambda x: x[0].upper() + x[1:] if type(x) is str else x)
    dms_count=len(dms_reference)
    print(f"There are {dms_count} DMSs in mapping file")
    
    with open(args.config_file) as f:
        config = json.load(f)
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/constants.json") as f:
        constants = json.load(f)
    uniprot_function_lookup = dms_reference[["UniProt_ID","coarse_selection_type"]]
    uniprot_function_lookup.columns = ["UniProt_ID", "Selection Type"]
    uniprot_neff_lookup = dms_reference[['UniProt_ID','MSA_Neff_L_category']].drop_duplicates()
    uniprot_neff_lookup.columns=['UniProt_ID','MSA_Neff_L_category']
    uniprot_taxon_lookup = dms_reference[['UniProt_ID','taxon']].drop_duplicates()
    uniprot_taxon_lookup.columns=['UniProt_ID','Taxon']
    if args.indel_mode:
        args.performance_by_depth = False

    if not os.path.isdir(args.output_performance_file_folder):
        os.mkdir(args.output_performance_file_folder)
    for metric in ['Spearman','AUC','MCC',"NDCG","Top_recall"]:
        if not os.path.isdir(args.output_performance_file_folder+os.sep+metric):
            os.mkdir(args.output_performance_file_folder+os.sep+metric)
    
    # Will populate model_types dynamically after scanning available model columns
    model_types = pd.DataFrame(columns=['Model type'])
    model_details = pd.DataFrame.from_dict(constants.get("model_details", {}), columns=['Model details'], orient='index')
    model_references = pd.DataFrame.from_dict(constants.get("model_references", {}), columns=['References'], orient='index')
    clean_names = constants.get("clean_names", {})
    performance_all_dms={}
    output_filename={}
    for metric in ['Spearman','AUC','MCC', "NDCG", "Top_recall"]:
        performance_all_dms[metric] = {}
        mutation_type = "substitutions" if not args.indel_mode else "indels"
        output_filename[metric] = "DMS_" + mutation_type + "_" + metric
        # Start with only metadata rows; model rows will be added dynamically via outer merges
        performance_all_dms[metric]['number_mutants'] = -1
        performance_all_dms[metric]["Selection Type"] = -1
        performance_all_dms[metric]["UniProt_ID"] = -1
        performance_all_dms[metric]['MSA_Neff_L_category'] = -1
        performance_all_dms[metric]['Taxon'] = -1
        performance_all_dms[metric] = pd.DataFrame.from_dict(performance_all_dms[metric], orient='index').reset_index()
        performance_all_dms[metric].columns = ['score', 'score_index']

    dms_ids = dms_reference["DMS_id"]
    all_models_found = set()
    for DMS_id in dms_ids:
        try:
            print(DMS_id)
            uniprot_id = dms_reference["UniProt_ID"][dms_reference["DMS_id"]==DMS_id].values[0]
            selection_type = dms_reference["coarse_selection_type"][dms_reference["DMS_id"]==DMS_id].values[0]
            msa_neff_category = dms_reference["MSA_Neff_L_category"][dms_reference["DMS_id"]==DMS_id].values[0]
            taxon = dms_reference["taxon"][dms_reference["DMS_id"]==DMS_id].values[0]
            if args.indel_mode:
                suffix = 'pll'
            else:
                suffix = args.scoring_method
            score_path = os.path.join(args.input_scoring_files_folder, f"{DMS_id}_zs_{suffix}.csv")
            if os.path.exists(score_path):
                merged_scores = pd.read_csv(score_path, dtype={'target_seq': str})
            if 'mutant' not in merged_scores:
                merged_scores['mutant'] = merged_scores['mutated_sequence'] #if mutant not in DMS file we default to mutated_sequence (eg., for indels)
            # Ensure binary labels exist for AUC/MCC if not provided
            if 'DMS_score_bin' not in merged_scores and 'DMS_score' in merged_scores:
                median_cutoff = merged_scores['DMS_score'].median()
                merged_scores['DMS_score_bin'] = (merged_scores['DMS_score'] >= median_cutoff).astype(int)
            if 'DMS_score' not in merged_scores:
                print(f"DMS_score column missing for {DMS_id}; skipping this DMS")
                continue
        except (KeyError, IndexError, NameError, ValueError, OSError):
            print(f"Scoring file for {DMS_id} missing")
            continue

        if not args.indel_mode and args.performance_by_depth:
            if 'mutant' in merged_scores:
                merged_scores['mutation_depth'] = merged_scores['mutant'].apply(lambda x: len(x.split(":")))
                merged_scores['mutation_depth_grouped'] = merged_scores['mutation_depth'].apply(lambda x: '5+' if x >= 5 else str(x))
            else:
                print(f"No 'mutant' or 'mutated_sequence' column to compute mutation depth for DMS {DMS_id}; setting to nan")
                merged_scores['mutation_depth_grouped'] = np.nan
        performance_dms = {}
        for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
            performance_dms[metric]={}
        # Determine available model score columns
        score_columns_present = [name for name in args.selected_model_names if name in merged_scores.columns]
        score_columns_present = list(dict.fromkeys(score_columns_present))
        all_models_found.update(score_columns_present)
        for score in score_columns_present:
            if score not in merged_scores:
                print(f"Model scores for {score} not in merged scores for DMS {DMS_id}")
                performance_dms["Spearman"][score] = np.nan
                performance_dms["AUC"][score] = np.nan
                performance_dms["MCC"][score] = np.nan
                performance_dms["NDCG"][score] = np.nan 
                performance_dms["Top_recall"][score] = np.nan
                continue
            performance_dms['Spearman'][score] = spearmanr(merged_scores['DMS_score'], merged_scores[score])[0]
            performance_dms["NDCG"][score] = calc_ndcg(merged_scores['DMS_score'], merged_scores[score])
            performance_dms["Top_recall"][score] = calc_toprecall(merged_scores['DMS_score'], merged_scores[score])
            try:
                performance_dms['AUC'][score] = roc_auc_score(y_true=merged_scores['DMS_score_bin'], y_score=merged_scores[score])
            except ValueError:
                print(f"AUC issue with: {DMS_id} for model: {score}")
                performance_dms['AUC'][score] = np.nan
            try:
                median_cutoff=merged_scores[score].median()
                merged_scores[score+"_bin"]=merged_scores[score].map(lambda x: 1 if x >= median_cutoff else 0)
                performance_dms['MCC'][score] = matthews_corrcoef(y_true=merged_scores['DMS_score_bin'], y_pred=merged_scores[score+"_bin"])
            except (KeyError, ValueError):
                print(f"MCC issue with: {DMS_id} for model: {score}")
                performance_dms['MCC'][score] = np.nan

        if not args.indel_mode and args.performance_by_depth:
            for score in score_columns_present:
                if score not in merged_scores:
                    print(f"Model scores for {score} not in merged scores for DMS {DMS_id}")
                    for depth in ['1','2','3','4','5+']:
                        performance_dms["Spearman"][score+'_'+depth] = np.nan
                        performance_dms["AUC"][score+'_'+depth] = np.nan
                        performance_dms["MCC"][score+'_'+depth] = np.nan 
                        performance_dms["NDCG"][score+'_'+depth] = np.nan
                        performance_dms["Top_recall"][score+'_'+depth] = np.nan
                    continue
                for depth in ['1','2','3','4','5+']:
                    merged_scores_depth = merged_scores[merged_scores.mutation_depth_grouped==depth]
                    if len(merged_scores_depth) > 0:
                        performance_dms['Spearman'][score+'_'+depth] = spearmanr(merged_scores_depth['DMS_score'], merged_scores_depth[score])[0]
                        performance_dms["NDCG"][score+'_'+depth] = calc_ndcg(merged_scores_depth['DMS_score'], merged_scores_depth[score])
                        performance_dms["Top_recall"][score+'_'+depth] = calc_toprecall(merged_scores_depth['DMS_score'], merged_scores_depth[score])
                        try:
                            performance_dms['AUC'][score+'_'+depth] = roc_auc_score(y_true=merged_scores_depth['DMS_score_bin'], y_score=merged_scores_depth[score])
                        except ValueError:
                            performance_dms['AUC'][score+'_'+depth] = np.nan
                        try:
                            performance_dms['MCC'][score+'_'+depth] = matthews_corrcoef(y_true=merged_scores_depth['DMS_score_bin'], y_pred=merged_scores_depth[score+"_bin"])
                        except (KeyError, ValueError):
                            performance_dms['MCC'][score+'_'+depth] = np.nan
                    else:
                        performance_dms['Spearman'][score+'_'+depth] = np.nan
                        performance_dms['AUC'][score+'_'+depth] = np.nan
                        performance_dms['MCC'][score+'_'+depth] = np.nan
                        performance_dms["NDCG"][score+'_'+depth] = np.nan
                        performance_dms["Top_recall"][score+'_'+depth] = np.nan
        print(f"Number of mutants: {len(merged_scores['DMS_score'].values)}")
        for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
            performance_dms[metric]['number_mutants']=len(merged_scores['DMS_score'].values)
            performance_dms[metric]['UniProt_ID'] = uniprot_id
            performance_dms[metric]["Selection Type"] = selection_type
            performance_dms[metric]['MSA_Neff_L_category'] = msa_neff_category
            performance_dms[metric]['Taxon'] = taxon
            performance_dms[metric] = pd.DataFrame.from_dict(performance_dms[metric],orient='index').reset_index()
            performance_dms[metric].columns=['score',DMS_id]
            performance_all_dms[metric]=pd.merge(performance_all_dms[metric],performance_dms[metric],on='score',how='outer')
    # Build model types dynamically
    try:
        if not args.indel_mode:
            cfg_models = config.get("model_list_zero_shot_substitutions_DMS", {})
        else:
            cfg_models = config.get("model_list_zero_shot_indels_DMS", {})
        model_types_dict = {}
        for model in sorted(all_models_found):
            model_types_dict[model] = cfg_models.get(model, {}).get("model_type", "Unknown")
        model_types = pd.DataFrame.from_dict(model_types_dict, columns=['Model type'], orient='index')
    except (AttributeError, ValueError):
        model_types = pd.DataFrame(columns=['Model type'])
    for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
        performance_all_dms[metric]=performance_all_dms[metric].set_index('score')
        del performance_all_dms[metric]['score_index']
        performance_all_dms[metric]=performance_all_dms[metric].transpose()
        for var in performance_all_dms[metric]:
            if var not in ['UniProt_ID','MSA_Neff_L_category','Taxon',"Selection Type"]:
                performance_all_dms[metric][var]=performance_all_dms[metric][var].astype(float).round(3)
            if var in ['number_mutants']:
                performance_all_dms[metric][var]=performance_all_dms[metric][var].astype(int)
        if not args.indel_mode and args.performance_by_depth:
            all_columns = performance_all_dms[metric].columns
            performance_all_dms_html=performance_all_dms[metric].copy()
            performance_all_dms_html.columns=performance_all_dms_html.columns.map(lambda x: clean_names.get(x, x))
            all_not_depth_columns = all_columns[[all_columns[x].split("_")[-1] not in ['1','2','3','4','5+'] for x in range(len(all_columns))]]
            all_not_depth_columns_clean = all_not_depth_columns.map(lambda x: clean_names.get(x, x))
            performance_all_dms_html[all_not_depth_columns_clean].to_html(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.html')
            DMS_perf_to_save = performance_all_dms[metric].copy()[all_not_depth_columns]
            DMS_perf_to_save.columns = DMS_perf_to_save.columns.map(lambda x: clean_names.get(x, x))
            DMS_perf_to_save.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.csv', index_label="DMS ID")
        else:
            performance_all_dms_html=performance_all_dms[metric].copy()
            performance_all_dms_html.columns = performance_all_dms_html.columns.map(lambda x: clean_names.get(x, x))
            performance_all_dms_html.to_html(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.html')
            DMS_perf_to_save = performance_all_dms[metric].copy()
            DMS_perf_to_save.columns = DMS_perf_to_save.columns.map(lambda x: clean_names.get(x, x))
            DMS_perf_to_save.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.csv', index_label="DMS ID")
        
        if not args.indel_mode:
            uniprot_metric_performance = performance_all_dms[metric].groupby(['UniProt_ID']).mean(numeric_only=True)
            uniprot_function_metric_performance = performance_all_dms[metric].groupby(['UniProt_ID',"Selection Type"]).mean(numeric_only=True)
            uniprot_metric_performance = uniprot_metric_performance.reset_index()
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_neff_lookup,on='UniProt_ID', how='left')
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_taxon_lookup,on='UniProt_ID', how='left')
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_function_lookup,on="UniProt_ID",how="left")
            del uniprot_metric_performance['number_mutants']
            del uniprot_function_metric_performance["number_mutants"]
            uniprot_level_average = uniprot_metric_performance.mean(numeric_only=True)
            uniprot_function_level_average = uniprot_function_metric_performance.groupby("Selection Type").mean(numeric_only=True)
            # bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance['TranceptEVE_L'],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_function_level_average = uniprot_function_level_average.reset_index()
            final_average = uniprot_function_level_average.mean(numeric_only=True) 
            if args.performance_by_depth:
                cols = [column for column in all_not_depth_columns if column not in ["number_mutants","Taxon","MSA_Neff_L_category","Selection Type","UniProt_ID"]]
                top_model = final_average.loc[cols].idxmax()
            else:
                top_model = final_average.idxmax()
            bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance[top_model],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_metric_performance.loc['Average'] = uniprot_level_average
            uniprot_function_level_average.loc['Average'] = final_average
            uniprot_metric_performance=uniprot_metric_performance.round(3)
            uniprot_function_level_average=uniprot_function_level_average.round(3)
            if args.performance_by_depth:
                uniprot_metric_performance[[column for column in all_not_depth_columns if column != "number_mutants"]].to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_Uniprot_level.csv', index=False)
                performance_by_depth = {}
                all_not_depth_columns = [x for x in all_not_depth_columns if x not in ['number_mutants',"UniProt_ID","MSA_Neff_L_category","Taxon"]]
                for depth in ['1','2','3','4','5+']:
                    depth_columns = all_columns[[all_columns[x].split("_")[-1]==depth for x in range(len(all_columns))]]
                    performance_by_depth[depth] = uniprot_function_metric_performance[depth_columns].mean(numeric_only=True).reset_index()
                    performance_by_depth[depth]['model_name'] = performance_by_depth[depth]['score'].map(lambda x: '_'.join(x.split('_')[:-1]))
                    performance_by_depth[depth]=performance_by_depth[depth][['model_name',0]]
                    performance_by_depth[depth].columns = ['model_name','Depth_'+depth]
                    performance_by_depth[depth].set_index('model_name', inplace=True)
                uniprot_function_level_average = uniprot_function_level_average[all_not_depth_columns]
            else:
                uniprot_metric_performance.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_Uniprot_level.csv', index=False)
            uniprot_function_level_average.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + "_Uniprot_Selection_Type_level.csv",index=False)
            if args.performance_by_depth:
                performance_by_msa_depth = performance_all_dms[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True)[[col for col in all_not_depth_columns if col != "Selection Type"]].transpose()
            else:
                performance_by_msa_depth = performance_all_dms[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True).transpose()
            performance_by_msa_depth = performance_by_msa_depth.reindex(columns=['Low','Medium','High'])
            performance_by_msa_depth.columns = ['Low_MSA_depth','Medium_MSA_depth','High_MSA_depth']
            if args.performance_by_depth:
                performance_by_taxon = performance_all_dms[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True)[[col for col in all_not_depth_columns if col != "Selection Type"]].transpose()
            else:
                performance_by_taxon = performance_all_dms[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True).transpose()
            performance_by_taxon = performance_by_taxon.reindex(columns=['Human','Eukaryote','Prokaryote','Virus'])
            performance_by_taxon.columns = ['Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus']
            performance_by_function = uniprot_function_level_average.drop(labels="Average",axis=0).set_index("Selection Type").transpose()
            performance_by_function.columns = ["Function_"+x for x in performance_by_function.columns]
            
            summary_performance = pd.merge(pd.DataFrame(final_average,columns=['Average_'+metric]), performance_by_msa_depth,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_taxon,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_function,left_index=True, right_index=True, how='inner')
            if args.performance_by_depth:
                for depth in ['1','2','3','4','5+']:
                    summary_performance = pd.merge(summary_performance, performance_by_depth[depth],left_index=True, right_index=True,how='inner')
            final_column_order = ['Model_name','Model type','Average_'+metric,'Bootstrap_standard_error_'+metric,'Function_Activity','Function_Binding','Function_Expression','Function_OrganismalFitness','Function_Stability','Low_MSA_depth','Medium_MSA_depth','High_MSA_depth','Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus','Depth_1','Depth_2','Depth_3','Depth_4','Depth_5+','Model details','References']

        else:
            performance_all_dms[metric].loc["Average"] = performance_all_dms[metric].mean(numeric_only=True)
            uniprot_metric_performance = performance_all_dms[metric].groupby(['UniProt_ID']).mean(numeric_only=True)
            uniprot_function_metric_performance = performance_all_dms[metric].groupby(['UniProt_ID',"Selection Type"]).mean(numeric_only=True)
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_function_lookup,on="UniProt_ID",how="left")
            del uniprot_metric_performance['number_mutants']
            uniprot_level_average = uniprot_metric_performance.mean(numeric_only=True)
            del uniprot_function_metric_performance["number_mutants"]
            uniprot_function_level_average = uniprot_function_metric_performance.groupby("Selection Type").mean(numeric_only=True)
            # bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance['TranceptEVE_M'],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_function_level_average = uniprot_function_level_average.reset_index()
            final_average = uniprot_function_level_average.mean(numeric_only=True) 
            top_model = final_average.idxmax()
            bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance[top_model],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_metric_performance.loc['Average'] = uniprot_level_average
            uniprot_function_level_average.loc['Average'] = final_average
            uniprot_metric_performance=uniprot_metric_performance.round(3)
            uniprot_function_level_average=uniprot_function_level_average.round(3)
            
            performance_by_msa_depth = performance_all_dms[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True).transpose()
            performance_by_msa_depth = performance_by_msa_depth.reindex(columns=['Low','Medium','High'])
            performance_by_msa_depth.columns = ['Low_MSA_depth','Medium_MSA_depth','High_MSA_depth']
            performance_by_taxon = performance_all_dms[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True).transpose()
            performance_by_taxon = performance_by_taxon.reindex(columns=['Human','Eukaryote','Prokaryote','Virus'])
            performance_by_taxon.columns = ['Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus']
            performance_by_function = uniprot_function_level_average.drop(labels="Average",axis=0).set_index("Selection Type").transpose()
            performance_by_function.columns = ["Function_"+x for x in performance_by_function.columns]
            
            summary_performance = pd.merge(pd.DataFrame(final_average,columns=['Average_'+metric]), performance_by_msa_depth,left_index=True,right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_taxon,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_function,left_index=True, right_index=True, how='inner')
            final_column_order = ['Model_name','Model type','Average_'+metric,'Bootstrap_standard_error_'+metric,'Function_Activity','Function_Binding','Function_Expression','Function_OrganismalFitness','Function_Stability','Low_MSA_depth','Medium_MSA_depth','High_MSA_depth','Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus','Model details','References']
        summary_performance.sort_values(by='Average_'+metric,ascending=False,inplace=True)
        summary_performance.index.name = 'Model_name'
        summary_performance.reset_index(inplace=True)
        summary_performance.index = range(1,len(summary_performance)+1)
        summary_performance.index.name = 'Model_rank'
        summary_performance = pd.merge(summary_performance, bootstrap_standard_error, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_types, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_details, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_references, left_on='Model_name', right_index=True, how='left')
        summary_performance=summary_performance.round(3)
        summary_performance['Model_name']=summary_performance['Model_name'].map(lambda x: clean_names.get(x, x))
        summary_performance=summary_performance.reindex(columns=final_column_order)
        summary_performance.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + 'Summary_performance_'+output_filename[metric]+'.csv')
        summary_performance.to_html(args.output_performance_file_folder + os.sep + metric + os.sep + 'Summary_performance_'+output_filename[metric]+'.html')

if __name__ == '__main__':
    main()
