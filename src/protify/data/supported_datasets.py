
supported_datasets = {
    'EC': 'GleghornLab/EC_reg',
    'GO-CC': 'GleghornLab/CC_reg',
    'GO-BP': 'GleghornLab/BP_reg',
    'GO-MF': 'GleghornLab/MF_reg',
    'MB': 'GleghornLab/MB_reg',
    'DeepLoc-2': 'GleghornLab/DL2_reg',
    'DeepLoc-10': 'GleghornLab/DL10_reg',
    'Subcellular': 'GleghornLab/SL_13',
    'enzyme-kcat': 'GleghornLab/enzyme_kcat',
    'solubility': 'GleghornLab/solubility_prediction',
    'localization': 'GleghornLab/localization_prediction',
    'temperature-stability': 'GleghornLab/temperature_stability',
    'peptide-HLA-MHC-affinity': 'GleghornLab/peptide_HLA_MHC_affinity_ppi',
    'optimal-temperature': 'GleghornLab/optimal_temperature',
    'optimal-ph': 'GleghornLab/optimal_ph',
    'optimal-ph-rigor': 'GleghornLab/optimal_ph_rigor',
    'material-production': 'GleghornLab/material_production',
    'fitness-prediction': 'GleghornLab/fitness_prediction',
    'number-of-folds': 'GleghornLab/fold_prediction',
    'cloning-clf': 'GleghornLab/cloning_clf',
    'stability-prediction': 'GleghornLab/stability_prediction',
    'human-ppi-saprot': 'GleghornLab/HPPI',
    'SecondaryStructure-3': 'GleghornLab/SS3',
    'SecondaryStructure-8': 'GleghornLab/SS8',
    'fluorescence-prediction': 'GleghornLab/fluorescence_prediction',
    'plastic': 'GleghornLab/plastic_degradation_benchmark',
    'gold-ppi': 'Synthyra/bernett_gold_ppi',
    'human-ppi-pinui': 'GleghornLab/HPPI_PiNUI',
    'yeast-ppi-pinui': 'GleghornLab/YPPI_PiNUI',
    'shs27-ppi-raw': 'Synthyra/SHS27k',
    'shs148-ppi-raw': 'Synthyra/SHS148k',
    'shs27-ppi-random': 'GleghornLab/ppi_SHS27k_random_2025',
    'shs148-ppi-random': 'GleghornLab/ppi_SHS148k_random_2025',
    'shs27-ppi-dfs': 'GleghornLab/ppi_SHS27k_dfs_2025',
    'shs148-ppi-dfs': 'GleghornLab/ppi_SHS148k_dfs_2025',
    'shs27-ppi-bfs': 'GleghornLab/ppi_SHS27k_bfs_2025',
    'shs148-ppi-bfs': 'GleghornLab/ppi_SHS148k_bfs_2025',
    'string-ppi-random': 'GleghornLab/ppi_STRING_random_2025',
    'string-ppi-dfs': 'GleghornLab/ppi_STRING_dfs_2025',
    'string-ppi-bfs': 'GleghornLab/ppi_STRING_bfs_2025',
    'plm-interact': 'GleghornLab/plm_interact_human_train_cross_ppi',
    'ppi-mutation-effect': 'GleghornLab/ppi_mutation_effect', # requires multi_column
    'PPA-ppi': 'Synthyra/ppi_affinity',
    'foldseek-fold': 'lhallee/foldseek_dataset', # prostt5
    'foldseek-inverse': 'lhallee/foldseek_dataset', # prostt5
    'ec-active': 'lhallee/ec_active',
    'proteingym_zs': 'proteingym_zs', # not a path, data loading for this is currently handled in benchmarks/proteingym/data_loader.py
    'proteingym_supervised': 'proteingym_supervised', # not a path, data loading for this is currently handled in benchmarks/proteingym/data_loader.py
    'taxon_domain': 'GleghornLab/taxonomy_domain_0.4_clusters',
    'taxon_kingdom': 'GleghornLab/taxonomy_kingdom_0.4_clusters',
    'taxon_phylum': 'GleghornLab/taxonomy_phylum_0.4_clusters',
    'taxon_class': 'GleghornLab/taxonomy_class_0.4_clusters',
    'taxon_order': 'GleghornLab/taxonomy_order_0.4_clusters',
    'taxon_family': 'GleghornLab/taxonomy_family_0.4_clusters',
    'taxon_genus': 'GleghornLab/taxonomy_genus_0.4_clusters',
    'taxon_species': 'GleghornLab/taxonomy_species_0.4_clusters',
    'diff_phylogeny': 'GleghornLab/diff_phylo',
    'plddt': 'GleghornLab/af2_plddt',
    'realness': 'GleghornLab/realness_dataset',
    'million_full': 'GleghornLab/millionfull_round_1_oct_2025',
    'soluprot': 'GleghornLab/soluprot',
    'ecoli_expression': 'GleghornLab/ecoli_expression',
    'KSMoFinder-clustered': 'GleghornLab/ksmo_clustered',
    'KSMoFinder': 'GleghornLab/KSmo_fixed',
    # bom-pooling paper (Hoang & Singh 2025) datasets
    'FLUO': 'GleghornLab/bom_fluorescence',
    'BLAC': 'GleghornLab/bom_blac',
    'remote_homology': 'GleghornLab/bom_remote_homology',
    'DPI': 'GleghornLab/bom_dpi',
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
    'Subcellular',
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
    'temperature-stability',
    'material-production',
    'fitness-prediction',
    'number-of-folds',
    'cloning-clf',
    'stability-prediction',
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
    'optimal-temperature',
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
    'optimal-temperature',
    'optimal-ph',
    'fitness-prediction',
]


vector_benchmark = [
    ### multi-label
    'EC', # SaProt
    # GO
    'GO-CC', # SaProt
    'GO-BP', # SaProt
    'GO-MF', # SaProt
    'Subcellular', # Custom - Tamar
    # ppi
    'shs27-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    'shs148-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    'string-ppi-bfs', # MGPPI + SHS27k + SHS148k + STRING
    ### classification
    'MB', # SaProt
    'DeepLoc-2', # SaProt
    'DeepLoc-10', # SaProt
    'solubility', # Biomap
    'temperature-stability', # Biomap
    'material-production', # Biomap
    'fitness-prediction', # Biomap
    'number-of-folds', # Biomap
    'cloning-clf', # Biomap
    'stability-prediction', # Biomap
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
    'optimal-temperature', # Biomap
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
    'SecondaryStructure-3', # tokenwise classification
]