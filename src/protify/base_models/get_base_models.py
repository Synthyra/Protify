from dataclasses import dataclass


currently_supported_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'Random',
    'Random-Transformer',
    'Random-ESM2-8',
    'Random-ESM2-35', # same as Random-Transformer
    'Random-ESM2-150',
    'Random-ESM2-650',
    'ESMC-300',
    'ESMC-600',
    'ESM2-diff-150',
    'ESM2-diffAV-150',
    'ProtBert',
    'ProtBert-BFD',
    'ProtT5',
    'ProtT5-XL-UniRef50-full-prec',
    'ProtT5-XXL-UniRef50',
    'ProtT5-XL-BFD',
    'ProtT5-XXL-BFD',
    'ANKH-Base',
    'ANKH-Large',
    'ANKH2-Large',
    'GLM2-150',
    'GLM2-650',
    'GLM2-GAIA',
    'DPLM-150',
    'DPLM-650',
    'DPLM-3B',
    'DSM-150',
    'DSM-650',
    'ProtCLM-1b'
]

standard_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'ESMC-300',
    'ESMC-600',
    'ProtBert',
    'ProtT5',
    'GLM2-150',
    'GLM2-650',
    'ANKH-Base',
    'ANKH-Large',
    'DPLM-150',
    'DPLM-650',
    'DSM-150',
    'DSM-650',
    'Random',
    'Random-Transformer',
]

experimental_models = []


@dataclass
class BaseModelArguments:
    def __init__(self, model_names: list[str] = None, **kwargs):
        if model_names[0] == 'standard':
            self.model_names = standard_models
        elif 'exp' in model_names[0].lower():
            self.model_names = experimental_models
        else:
            self.model_names = model_names


def get_base_model(model_name: str):
    if 'random' in model_name.lower():
        from .random import build_random_model
        return build_random_model(model_name)
    elif 'esm2' in model_name.lower() or 'dsm' in model_name.lower():
        from .esm2 import build_esm2_model
        return build_esm2_model(model_name)
    elif 'esmc' in model_name.lower():
        from .esmc import build_esmc_model
        return build_esmc_model(model_name)
    elif 'protbert' in model_name.lower():
        from .protbert import build_protbert_model
        return build_protbert_model(model_name)
    elif 'prott5' in model_name.lower():
        from .prott5 import build_prott5_model
        return build_prott5_model(model_name)
    elif 'ankh' in model_name.lower():
        from .ankh import build_ankh_model
        return build_ankh_model(model_name)
    elif 'glm' in model_name.lower():
        from .glm import build_glm2_model
        return build_glm2_model(model_name)
    elif 'dplm' in model_name.lower():
        from .dplm import build_dplm_model
        return build_dplm_model(model_name)
    elif 'protclm' in model_name.lower():
        from .protCLM import build_protCLM
        return build_protCLM(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_base_model_for_training(model_name: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    if 'esm2' in model_name.lower() or 'dsm' in model_name.lower():
        from .esm2 import get_esm2_for_training
        return get_esm2_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'esmc' in model_name.lower():
        from .esmc import get_esmc_for_training
        return get_esmc_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'protbert' in model_name.lower():
        from .protbert import get_protbert_for_training
        return get_protbert_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'prott5' in model_name.lower():
        from .prott5 import get_prott5_for_training
        return get_prott5_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'ankh' in model_name.lower():
        from .ankh import get_ankh_for_training
        return get_ankh_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'glm' in model_name.lower():
        from .glm import get_glm2_for_training
        return get_glm2_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'dplm' in model_name.lower():
        from .dplm import get_dplm_for_training
        return get_dplm_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'protclm' in model_name.lower():
        from .protCLM import get_protCLM_for_training
        return get_protCLM_for_training(model_name, tokenwise, num_labels, hybrid)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_tokenizer(model_name: str):
    if 'esm2' in model_name.lower() or 'random' in model_name.lower() or 'dsm' in model_name.lower():
        from .esm2 import get_esm2_tokenizer
        return get_esm2_tokenizer(model_name)
    elif 'esmc' in model_name.lower():
        from .esmc import get_esmc_tokenizer
        return get_esmc_tokenizer(model_name)
    elif 'protbert' in model_name.lower():
        from .protbert import get_protbert_tokenizer
        return get_protbert_tokenizer(model_name)
    elif 'prott5' in model_name.lower():
        from .prott5 import get_prott5_tokenizer
        return get_prott5_tokenizer(model_name)
    elif 'ankh' in model_name.lower():
        from .ankh import get_ankh_tokenizer
        return get_ankh_tokenizer(model_name)
    elif 'glm' in model_name.lower():
        from .glm import get_glm2_tokenizer
        return get_glm2_tokenizer(model_name)
    elif 'dplm' in model_name.lower():
        from .dplm import get_dplm_tokenizer
        return get_dplm_tokenizer(model_name)
    elif 'protclm' in model_name.lower():
        from .protCLM import get_protCLM_tokenizer
        return get_protCLM_tokenizer(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == '__main__':
    # py -m src.protify.base_models.get_base_models
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and list supported models')
    parser.add_argument('--download', action='store_true', help='Download all standard models')
    parser.add_argument('--list', action='store_true', help='List all supported models with descriptions')
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    if args.list:
        try:
            from resource_info import model_descriptions
            print("\n=== Currently Supported Models ===\n")
            
            max_name_len = max(len(name) for name in currently_supported_models)
            max_type_len = max(len(model_descriptions.get(name, {}).get('type', 'Unknown')) for name in currently_supported_models if name in model_descriptions)
            max_size_len = max(len(model_descriptions.get(name, {}).get('size', 'Unknown')) for name in currently_supported_models if name in model_descriptions)
            
            # Print header
            print(f"{'Model':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Size':<{max_size_len+2}}Description")
            print("-" * (max_name_len + max_type_len + max_size_len + 50))
            
            for model_name in currently_supported_models:
                if model_name in model_descriptions:
                    model_info = model_descriptions[model_name]
                    print(f"{model_name:<{max_name_len+2}}{model_info.get('type', 'Unknown'):<{max_type_len+2}}{model_info.get('size', 'Unknown'):<{max_size_len+2}}{model_info.get('description', 'No description available')}")
                else:
                    print(f"{model_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_size_len+2}}No description available")
                    
            print("\n=== Standard Models ===\n")
            for model_name in standard_models:
                print(f"- {model_name}")
                
        except ImportError:
            print("Model descriptions file not found. Only listing model names.")
            print("\n=== Currently Supported Models ===\n")
            for model_name in currently_supported_models:
                print(f"- {model_name}")
                
            print("\n=== Standard Models ===\n")
            for model_name in standard_models:
                print(f"- {model_name}")
    
    if args.download:
        ### This will download all standard models
        from torchinfo import summary
        from ..utils import clear_screen
        download_args = BaseModelArguments(model_names=['standard'])
        for model_name in download_args.model_names:
            model, tokenizer = get_base_model(model_name)
            print(f'Downloaded {model_name}')
            tokenized = tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', return_tensors='pt').input_ids
            summary(model, input_data=tokenized)
            clear_screen()
