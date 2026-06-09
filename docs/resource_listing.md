# Listing Supported Models and Datasets

Protify provides several ways to view and explore the supported models and datasets. This documentation explains how to use these features.

## Using the README Toggle Sections

The main README.md file contains expandable toggle sections for both models and datasets:

- **Currently Supported Models**: Click the toggle to expand and see a complete table of models with their descriptions, sizes, and types.
- **Currently Supported Datasets**: Click the toggle to expand and see a complete table of datasets with their descriptions, types, and tasks.

## Command-Line Listing

Protify provides command-line utilities for listing models and datasets with detailed information:

### Listing Models

To list all supported models with their descriptions:

```bash
# List all supported models
py -m src.protify.base_models.get_base_models --list

# To download standard models
py -m src.protify.base_models.get_base_models --download
```

### Listing Datasets

To list all supported datasets with their descriptions:

```bash
# List all datasets
py -m src.protify.data.dataset_utils --list

# Get information about a specific dataset
py -m src.protify.data.dataset_utils --info EC
```

### Combined Listing

For a combined view of both models and datasets:

```bash
# List both models and datasets
py -m src.protify.resource_info --all

# List only standard models and datasets
py -m src.protify.resource_info --all --standard-only

# List only models
py -m src.protify.resource_info --models

# List only datasets
py -m src.protify.resource_info --datasets
```

## Programmatic Access

You can also access model and dataset information programmatically:

```python
# For models
from src.protify.resource_info import model_descriptions
from src.protify.base_models.get_base_models import currently_supported_models, standard_models

# Get information about a specific model
model_info = model_descriptions.get('ESM2-150', {})
print(f"Model: ESM2-150")
print(f"Description: {model_info.get('description', 'N/A')}")
print(f"Size: {model_info.get('size', 'N/A')}")
print(f"Type: {model_info.get('type', 'N/A')}")

# For datasets
from src.protify.resource_info import dataset_descriptions
from src.protify.data.supported_datasets import supported_datasets

# Get information about a specific dataset
dataset_info = dataset_descriptions.get('EC', {})
print(f"Dataset: EC")
print(f"Description: {dataset_info.get('description', 'N/A')}")
print(f"Type: {dataset_info.get('type', 'N/A')}")
print(f"Task: {dataset_info.get('task', 'N/A')}")
```

## Model Group Types

Models in Protify are generally grouped into the following categories:

1. **Protein Language Models**: Pre-trained models that have learned protein properties from large-scale sequence data (e.g., ESM2, ProtBert)
2. **Baseline Controls**: Models with random weights for comparison (e.g., Random, Random-Transformer)

`ESMC-6B` is available as a supported preset but is intentionally excluded from the standard model list to avoid accidental 6B benchmark runs.

## Dataset Group Types

Datasets are categorized by their task types:

1. **Multi-label Classification**: Datasets where each protein can have multiple labels (e.g., EC, GO-CC)
2. **Classification**: Binary or multi-class classification tasks (e.g., DeepLoc-2, DeepLoc-10)
3. **Regression**: Prediction of continuous values (e.g., enzyme-kcat, optimal-temperature)
4. **Protein-Protein Interaction**: Tasks focused on protein interactions (e.g., human-ppi, gold-ppi)
5. **Token-wise Classification/Regression**: Residue-level prediction tasks (e.g., SecondaryStructure-3)

---

## See also

- [Documentation hub](README.md) for the full documentation index
- [Data](data.md) for how datasets are loaded and used
- [Models and embeddings](models_and_embeddings.md) for base models and embedding options
