"""
Dataset descriptions derived from the README "Currently Supported Datasets" table.

Each entry provides:
- type: One of {BC, MCC, MLC, SLC, R, TR, Various}
- task: Short task description
- description: Brief dataset blurb
- tokenwise: Whether the task is token-wise (per-residue)
- multiple_inputs: Whether the dataset involves multiple sequence inputs per sample
"""

dataset_descriptions = {
    # Multi-label classification (MLC)
    "EC": {
        "type": "MLC",
        "task": "Protein function prediction",
        "description": "Enzyme Commission numbers dataset for predicting enzyme function classification.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "GO-CC": {
        "type": "MLC",
        "task": "Protein localization prediction",
        "description": "Gene Ontology Cellular Component dataset for predicting protein localization in cells.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "GO-BP": {
        "type": "MLC",
        "task": "Protein function prediction",
        "description": "Gene Ontology Biological Process dataset for predicting protein involvement in biological processes.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "GO-MF": {
        "type": "MLC",
        "task": "Protein function prediction",
        "description": "Gene Ontology Molecular Function dataset for predicting protein molecular functions.",
        "tokenwise": False,
        "multiple_inputs": False,
    },

    # Binary classification (BC)
    "MB": {
        "type": "BC",
        "task": "Protein-metal binding prediction",
        "description": "Metal ion binding dataset for predicting protein-metal interactions.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "DeepLoc-2": {
        "type": "BC",
        "task": "Protein localization prediction",
        "description": "Binary classification dataset for predicting protein localization in 2 categories.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "solubility": {
        "type": "BC",
        "task": "Protein solubility prediction",
        "description": "Dataset for predicting protein solubility properties.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "temperature-stability": {
        "type": "BC",
        "task": "Protein stability prediction",
        "description": "Dataset for predicting protein stability at different temperatures.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "material-production": {
        "type": "BC",
        "task": "Protein application prediction",
        "description": "Dataset for predicting protein suitability for material production.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "fitness-prediction": {
        "type": "BC",
        "task": "Protein fitness prediction",
        "description": "Dataset for predicting protein fitness in various environments.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "number-of-folds": {
        "type": "BC",
        "task": "Protein structure prediction",
        "description": "Dataset for predicting the number of structural folds in proteins.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "cloning-clf": {
        "type": "BC",
        "task": "Protein engineering prediction",
        "description": "Dataset for predicting protein suitability for cloning operations.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "stability-prediction": {
        "type": "BC",
        "task": "Protein stability prediction",
        "description": "Dataset for predicting overall protein stability.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "plastic": {
        "type": "BC",
        "task": "Enzyme function prediction",
        "description": "Dataset for predicting protein capability for plastic degradation.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "realness": {
        "type": "BC",
        "task": "Authenticity prediction",
        "description": "Protein realness dataset.",
        "tokenwise": False,
        "multiple_inputs": False,
    },

    # Multi-class classification (MCC)
    "DeepLoc-10": {
        "type": "MCC",
        "task": "Protein localization prediction",
        "description": "Multi-class classification dataset for predicting protein localization in 10 categories.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "Subcellular": {
        "type": "MCC",
        "task": "Protein localization prediction",
        "description": "Dataset for predicting subcellular localization of proteins.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "localization": {
        "type": "MCC",
        "task": "Protein localization prediction",
        "description": "Dataset for predicting subcellular localization of proteins.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "foldseek-fold": {
        "type": "MCC",
        "task": "Protein structure prediction",
        "description": "Dataset for protein fold classification using Foldseek.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "foldseek-inverse": {
        "type": "MCC",
        "task": "Protein structure prediction",
        "description": "Inverse protein fold prediction dataset.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "ec-active": {
        "type": "MCC",
        "task": "Enzyme function prediction",
        "description": "Dataset for predicting active enzyme classes.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_domain": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at domain level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_kingdom": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at kingdom level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_phylum": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at phylum level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_class": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at class level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_order": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at order level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_family": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at family level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_genus": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at genus level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "taxon_species": {
        "type": "MCC",
        "task": "Taxonomic prediction",
        "description": "Taxonomic classification at species level.",
        "tokenwise": False,
        "multiple_inputs": False,
    },

    # Regression (R)
    "enzyme-kcat": {
        "type": "R",
        "task": "Enzyme kinetics prediction",
        "description": "Dataset for predicting enzyme catalytic rate constants (kcat).",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "optimal-temperature": {
        "type": "R",
        "task": "Protein property prediction",
        "description": "Dataset for predicting the optimal temperature for protein function.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "optimal-ph": {
        "type": "R",
        "task": "Protein property prediction",
        "description": "Dataset for predicting the optimal pH for protein function.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "optimal-ph-rigor": {
        "type": "R",
        "task": "Protein property prediction",
        "description": "Rigorously curated variant of the optimal pH dataset for predicting the optimal pH for protein function.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
    "fluorescence-prediction": {
        "type": "R",
        "task": "Protein property prediction",
        "description": "Dataset for predicting protein fluorescence properties.",
        "tokenwise": True,
        "multiple_inputs": False,
    },
    "PPA-ppi": {
        "type": "R",
        "task": "protein-protein affinity prediction",
        "description": "Protein-Protein Affinity dataset from Bindwell.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "million_full": {
        "type": "R",
        "task": "Protein fitness prediction",
        "description": "Large-scale enzyme variant dataset, from Millionfull preprint October 2025",
        "tokenwise": False,
        "multiple_inputs": False,
    },

    # Single-label classification (SLC) and PPI
    "gold-ppi": {
        "type": "SLC",
        "task": "PPI prediction",
        "description": "Gold standard dataset for protein-protein interaction prediction.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "human-ppi-saprot": {
        "type": "SLC",
        "task": "PPI prediction",
        "description": "Human protein-protein interaction dataset from SAProt paper.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "human-ppi-pinui": {
        "type": "SLC",
        "task": "PPI prediction",
        "description": "Human protein-protein interaction dataset from PiNUI.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "yeast-ppi-pinui": {
        "type": "SLC",
        "task": "PPI prediction",
        "description": "Yeast protein-protein interaction dataset from PiNUI.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "peptide-HLA-MHC-affinity": {
        "type": "SLC",
        "task": "Binding affinity prediction",
        "description": "Dataset for predicting peptide binding affinity to HLA/MHC complexes.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "ppi-mutation-effect": {
        "type": "SLC",
        "task": "PPI effect prediction",
        "description": "Compare wild type, mutated, and target sequence to determine if PPI is stronger or not.",
        "tokenwise": False,
        "multiple_inputs": True,
    },

    # PPI sets (MLC variants and splits)
    "shs27-ppi-raw": {
        "type": "SLC",
        "task": "PPI type prediction",
        "description": "Raw SHS27k with single-label labels.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs148-ppi-raw": {
        "type": "SLC",
        "task": "PPI type prediction",
        "description": "Raw SHS148k with single-label labels.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs27-ppi-random": {
        "type": "MLC",
        "task": "PPI prediction",
        "description": "SHS27k",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs148-ppi-random": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "SHS148k CD-Hit 40%, multi-label lables, randomized data splits.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs27-ppi-dfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "SHS27k CD-Hit 40%, multi-label lables, data splits via depth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs148-ppi-dfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "SHS148k CD-Hit 40%, multi-label lables, data splits via depth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs27-ppi-bfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "SHS27k CD-Hit 40%, multi-label lables, data splits via breadth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "shs148-ppi-bfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "SHS148k CD-Hit 40%, multi-label lables, data splits via breadth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "string-ppi-random": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "STRING CD-Hit 40%, multi-label lables, randomized data splits.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "string-ppi-dfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "STRING CD-Hit 40%, multi-label lables, data splits via depth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },
    "string-ppi-bfs": {
        "type": "MLC",
        "task": "PPI type prediction",
        "description": "STRING CD-Hit 40%, multi-label lables, data splits via breadth first search.",
        "tokenwise": False,
        "multiple_inputs": True,
    },

    # Token-wise tasks
    "SecondaryStructure-3": {
        "type": "MCC",
        "task": "Protein structure prediction",
        "description": "Dataset for predicting protein secondary structure in 3 classes.",
        "tokenwise": True,
        "multiple_inputs": False,
    },
    "SecondaryStructure-8": {
        "type": "MCC",
        "task": "Protein structure prediction",
        "description": "Dataset for predicting protein secondary structure in 8 classes.",
        "tokenwise": True,
        "multiple_inputs": False,
    },
    "plddt": {
        "type": "TR",
        "task": "Confidence prediction",
        "description": "AlphaFold pLDDT confidence score prediction.",
        "tokenwise": True,
        "multiple_inputs": False,
    },

    # Misc
    "diff_phylogeny": {
        "type": "Various",
        "task": "Phylogeny prediction",
        "description": "Differential phylogeny dataset.",
        "tokenwise": False,
        "multiple_inputs": False,
    },
}


