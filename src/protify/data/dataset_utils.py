from .supported_datasets import (
    dataset_aliases,
    get_dataset_source,
    internal_datasets,
    possible_with_vector_reps,
    resolve_dataset_name,
    standard_data_benchmark,
    supported_datasets,
    testing,
)


def list_supported_datasets(with_descriptions: bool = True) -> None:
    """
    Lists all supported datasets with optional descriptions.
    
    Args:
        with_descriptions (bool): Whether to include descriptions (if available)
    """
    try:
        from .dataset_descriptions import dataset_descriptions
        has_descriptions = True
    except ImportError:
        has_descriptions = False
        
    if not with_descriptions or not has_descriptions:
        print("\n=== Supported Datasets ===\n")
        for dataset_name in supported_datasets:
            print(f"- {dataset_name}: {supported_datasets[dataset_name]}")
        return
    
    print("\n=== Supported Datasets ===\n")
    
    # Calculate maximum widths for formatting
    max_name_len = max(len(name) for name in supported_datasets)
    max_type_len = max(len(dataset_descriptions.get(name, {}).get('type', 'Unknown')) for name in supported_datasets if name in dataset_descriptions)
    max_task_len = max(len(dataset_descriptions.get(name, {}).get('task', 'Unknown')) for name in supported_datasets if name in dataset_descriptions)
    
    # Print header
    print(f"{'Dataset':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Task':<{max_task_len+2}}Description")
    print("-" * (max_name_len + max_type_len + max_task_len + 50))
    
    # Print dataset information
    for dataset_name in supported_datasets:
        if dataset_name in dataset_descriptions:
            dataset_info = dataset_descriptions[dataset_name]
            print(f"{dataset_name:<{max_name_len+2}}{dataset_info.get('type', 'Unknown'):<{max_type_len+2}}{dataset_info.get('task', 'Unknown'):<{max_task_len+2}}{dataset_info.get('description', 'No description available')}")
        else:
            print(f"{dataset_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_task_len+2}}No description available")
    
    print("\n=== Standard Benchmark Datasets ===\n")
    for dataset_name in standard_data_benchmark:
        print(f"- {dataset_name}")


def get_dataset_info(dataset_name: str):
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dataset information or None if not found
    """
    canonical_name = resolve_dataset_name(dataset_name)

    try:
        from .dataset_descriptions import dataset_descriptions
        if canonical_name in dataset_descriptions:
            dataset_info = dict(dataset_descriptions[canonical_name])
            if dataset_name in dataset_aliases:
                dataset_info['name'] = canonical_name
                dataset_info['alias'] = dataset_name
            return dataset_info
    except ImportError:
        pass
        
    if canonical_name in supported_datasets:
        return {"name": canonical_name, "source": get_dataset_source(canonical_name)}
    
    return None


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='List and describe supported datasets')
    parser.add_argument('--list', action='store_true', help='List all supported datasets')
    parser.add_argument('--info', type=str, help='Get information about a specific dataset')
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    if args.list:
        list_supported_datasets()
        
    if args.info:
        dataset_info = get_dataset_info(args.info)
        if dataset_info:
            print(f"\n=== Dataset: {args.info} ===\n")
            for key, value in dataset_info.items():
                print(f"{key.capitalize()}: {value}")
        else:
            print(f"Dataset '{args.info}' not found in supported datasets.") 