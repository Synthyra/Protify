
supported_datasets = {
    'EC': 'GleghornLab/EC_reg',
    'GO-CC': 'GleghornLab/CC_reg',
    'GO-BP': 'GleghornLab/BP_reg',
    'GO-MF': 'GleghornLab/MF_reg',
    'SL13': 'GleghornLab/SL_13',
    'shs27-ppi-bfs': 'GleghornLab/ppi_SHS27k_bfs_2025',
    'shs148-ppi-bfs': 'GleghornLab/ppi_SHS148k_bfs_2025',
    'string-ppi-bfs': 'GleghornLab/ppi_STRING_bfs_2025',
    'DeepLoc-10': 'GleghornLab/DL10_reg',
    'ec-active': 'lhallee/ec_active',
    'taxon_species': 'GleghornLab/taxonomy_species_0.4_clusters',
    'number-of-folds': 'GleghornLab/fold_prediction',
    'MB': 'GleghornLab/MB_reg',
    'DeepLoc-2': 'GleghornLab/DL2_reg',
    'solubility': 'GleghornLab/solubility_prediction',
    'diff_phylogeny': 'GleghornLab/diff_phylo',
    'temp-stability': 'GleghornLab/temperature_stability',
    'mat-production': 'GleghornLab/material_production',
    'cloning-clf': 'GleghornLab/cloning_clf',
    'soluprot': 'GleghornLab/soluprot',
    'plm-interact': 'GleghornLab/plm_interact_human_train_cross_ppi',
    'gold-ppi': 'Synthyra/bernett_gold_ppi',
    'ecoli_expression': 'GleghornLab/ecoli_expression',
    'KSMoFinder': 'GleghornLab/KSmo_fixed',
    'KSMoFinder-clustered': 'GleghornLab/ksmo_clustered',
    'fitness-pred': 'GleghornLab/fitness_prediction',
    'stability-pred': 'GleghornLab/stability_prediction',
    'enzyme-kcat': 'GleghornLab/enzyme_kcat',
    'opt-temp': 'GleghornLab/optimal_temperature',
    'optimal-ph': 'GleghornLab/optimal_ph',
    'million_full': 'GleghornLab/millionfull_round_1_oct_2025',
    'PPA-ppi': 'Synthyra/ppi_affinity',
    'SS3': 'GleghornLab/SS3',
    'SS8': 'GleghornLab/SS8',
    'foldseek-fold': 'lhallee/foldseek_dataset', # prostt5
    'foldseek-inverse': 'lhallee/foldseek_dataset', # prostt5
    'plddt': 'GleghornLab/af2_plddt',
    'shs27-ppi-random': 'GleghornLab/ppi_SHS27k_random_2025',
    'shs27-ppi-dfs': 'GleghornLab/ppi_SHS27k_dfs_2025',
    'shs148-ppi-random': 'GleghornLab/ppi_SHS148k_random_2025',
    'shs148-ppi-dfs': 'GleghornLab/ppi_SHS148k_dfs_2025',
    'string-ppi-random': 'GleghornLab/ppi_STRING_random_2025',
    'string-ppi-dfs': 'GleghornLab/ppi_STRING_dfs_2025',
    'shs27-ppi-raw': 'Synthyra/SHS27k',
    'shs148-ppi-raw': 'Synthyra/SHS148k',
    'localization': 'GleghornLab/localization_prediction',
    'taxon_domain': 'GleghornLab/taxonomy_domain_0.4_clusters',
    'taxon_kingdom': 'GleghornLab/taxonomy_kingdom_0.4_clusters',
    'taxon_phylum': 'GleghornLab/taxonomy_phylum_0.4_clusters',
    'taxon_class': 'GleghornLab/taxonomy_class_0.4_clusters',
    'taxon_order': 'GleghornLab/taxonomy_order_0.4_clusters',
    'taxon_family': 'GleghornLab/taxonomy_family_0.4_clusters',
    'taxon_genus': 'GleghornLab/taxonomy_genus_0.4_clusters',
    'peptide-HLA-MHC': 'GleghornLab/peptide_HLA_MHC_affinity_ppi',
    'human-ppi-saprot': 'GleghornLab/HPPI',
    'human-ppi-pinui': 'GleghornLab/HPPI_PiNUI',
    'yeast-ppi-pinui': 'GleghornLab/YPPI_PiNUI',
    'ppi-mutation-effect': 'GleghornLab/ppi_mutation_effect', # requires multi_column
    'bernett_processed': 'lhallee/bernett_processed',
    'fluorescence': 'GleghornLab/fluorescence_prediction',
    #additional, not in table
    'plastic': 'GleghornLab/plastic_degradation_benchmark',
    'proteingym_zs': 'proteingym_zs', # not a path, data loading for this is currently handled in benchmarks/proteingym/data_loader.py
    'proteingym_supervised': 'proteingym_supervised', # not a path, data loading for this is currently handled in benchmarks/proteingym/data_loader.py
    'realness': 'GleghornLab/realness_dataset',
}

internal_datasets = {
    'plastic': 'GleghornLab/plastic_degradation_benchmark',
}

# TODO update
possible_with_vector_reps = [
    ### multi-label
    'EC',
    # GO
    'GO-CC',
    'GO-BP',
    'GO-MF',
    'SL13',
    # ppi
    'shs27-ppi-random',
    'shs27-ppi-dfs',
    'shs27-ppi-bfs',
    'shs148-ppi-random',
    'shs148-ppi-dfs',
    'shs148-ppi-bfs',
    'string-ppi-random',
    'string-ppi-dfs',
    'string-ppi-bfs',
    ### classification
    'MB',
    'DeepLoc-2',
    'DeepLoc-10',
    'solubility',
    'temp-stability',
    'mat-production',
    'fitness-pred',
    'number-of-folds',
    'cloning-clf',
    'stability-pred',
    'ec-active',
    'localization',
    # taxonomy
    'taxon_domain',
    'taxon_kingdom',
    'taxon_phylum',
    'taxon_class',
    'taxon_order',
    'taxon_family',
    'taxon_genus',
    'taxon_species',
    'diff_phylogeny',
    # ppi
    'shs27-ppi-raw',
    'shs148-ppi-raw',
    'plm-interact',
    'gold-ppi',
    'string-ppi-bfs',
    'human-ppi-saprot',
    'human-ppi-pinui',
    'yeast-ppi-pinui',
    ### regression
    'enzyme-kcat',
    'opt-temp',
    'optimal-ph',
    'million_full',
    # ppi
    'PPA-ppi',
]

# TODO update
standard_data_benchmark = [
    'ec-active',
    'EC',
    'GO-CC',
    'GO-BP',
    'GO-MF',
    'MB',
    'DeepLoc-2',
    'DeepLoc-10',
    'enzyme-kcat',
    'opt-temp',
    'optimal-ph',
    'fitness-pred',
]


vector_benchmark = [
    ### multi-label
    'EC', # SaProt
    # GO
    'GO-CC', # SaProt
    'GO-BP', # SaProt
    'GO-MF', # SaProt
    'SL13', # Custom - Tamar
    # ppi
    'shs27-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    'shs148-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    'string-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    ### classification
    'MB', # SaProt
    'DeepLoc-2', # SaProt
    'DeepLoc-10', # SaProt
    'solubility', # Biomap
    'temp-stability', # Biomap
    'mat-production', # Biomap
    'fitness-pred', # Biomap
    'number-of-folds', # Biomap
    'cloning-clf', # Biomap
    'stability-pred', # Biomap
    'ec-active', # Custom - Logan
    'soluprot', # Custom - SoluProt
    # taxonomy
    'taxon_species', # Accidental taxonomists
    'diff_phylogeny', # Accidental taxonomists
    # ppi
    'plm-interact', # PLM-Interact
    'gold-ppi', # Bernett
    ### regression
    'enzyme-kcat', # Custom - Logan - Biomap
    'opt-temp', # Biomap
    'optimal-ph', # Biomap
    'million_full', # Millionfull
    # ppi
    'PPA-ppi', # Custom - Logan - Bindwell
]


testing = [
    'EC', # multilabel
    'DeepLoc-2', # singlelabel
    'DeepLoc-10', # multiclass
    'enzyme-kcat', # regression
    'human-ppi', # ppi
    'plddt', # tokenwise regression
    'SS3', # tokenwise classification
]
