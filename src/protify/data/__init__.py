from .supported_datasets import (
    dataset_aliases,
    get_dataset_source,
    internal_datasets,
    possible_with_vector_reps,
    resolve_dataset_name,
    rigorous_benchmark,
    standard_data_benchmark,
    supported_datasets,
    testing,
)


try:
    from .dataset_descriptions import dataset_descriptions
except ImportError:
    dataset_descriptions = {}

from .dataset_utils import get_dataset_info, list_supported_datasets

__all__ = [
    "dataset_aliases",
    "dataset_descriptions",
    "get_dataset_info",
    "get_dataset_source",
    "internal_datasets",
    "list_supported_datasets",
    "possible_with_vector_reps",
    "resolve_dataset_name",
    "rigorous_benchmark",
    "standard_data_benchmark",
    "supported_datasets",
    "testing",
]
