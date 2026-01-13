import entrypoint_setup

import os
import argparse
import yaml
from types import SimpleNamespace


def parse_arguments():  
    parser = argparse.ArgumentParser(description="Script with arguments mirroring the provided YAML settings.")
    # ----------------- ID ----------------- #
    parser.add_argument("--hf_username", default="Synthyra", help="Hugging Face username.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token.")
    parser.add_argument("--synthyra_api_key", default=None, help="Synthyra API key.")
    parser.add_argument("--wandb_api_key", default=None, help="Wandb API key.")

    # ----------------- Paths ----------------- #
    parser.add_argument("--hf_home", type=str, default=None, help="Customize the HF cache directory.")
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to the YAML file.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to the log directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to the results directory.")
    parser.add_argument("--model_save_dir", default="weights", help="Directory to save models.")
    parser.add_argument("--embedding_save_dir", default="embeddings", help="Directory to save embeddings.")
    parser.add_argument("--download_dir", default="Synthyra/vector_embeddings", help="Directory to download embeddings to.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots.")
    parser.add_argument("--replay_path", type=str, default=None, help="Path to the replay file.")
    parser.add_argument("--pretrained_probe_path", type=str, default=None) # TODO not used right now
    
    # ----------------- DataArguments ----------------- #
    parser.add_argument("--delimiter", default=",", help="Delimiter for data.")
    parser.add_argument("--col_names", nargs="+", default=["seqs", "labels"], help="Column names.") # DEPRECATED, found automatically now
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--trim", action="store_true", default=False,
                        help="Whether to trim sequences (default: False). If False, sequences are removed from the dataset if they are longer than max length. If True, they are truncated to max length."
                        )
    parser.add_argument("--data_names", nargs="+", default=[], help="List of HF dataset names.") # TODO rename to data_names
    parser.add_argument("--data_dirs", nargs="+", default=[], help="List of local data directories.")

    # ----------------- BaseModelArguments ----------------- #
    parser.add_argument("--model_names", nargs="+", default=["ESM2-8"], help="List of model names to use. To use a custom model, use the format 'custom---<path_to_model>'.")

    # ----------------- ProbeArguments ----------------- #
    parser.add_argument("--probe_type", choices=["linear", "transformer", "retrievalnet", "lyra"], default="linear", help="Type of probe.")
    parser.add_argument("--tokenwise", action="store_true", default=False, help="Tokenwise probe (default: False).")
    ### TODO refactor to hidden_size
    parser.add_argument("--hidden_size", type=int, default=8192, help="Hidden dimension size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--pre_ln", action="store_false", default=True,
                        help="Disable pre-layernorm (default: enabled). Use --pre_ln to toggle off.")
    parser.add_argument("--classifier_size", type=int, default=4096, help="Feed-forward dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for the transformer layers.")
    parser.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout rate for the classifier.")
    parser.add_argument("--head_size", type=int, default=64, help="Attention head size. The number of attention heads is computed as hidden_size // head_size.")
    parser.add_argument("--rotary", action="store_false", default=True,
                        help="Disable rotary embeddings (default: enabled). Use --rotary to toggle off.")
    parser.add_argument("--probe_pooling_types", nargs="+", default=["mean", "var"], help="Pooling types to use.")
    parser.add_argument("--save_model", action="store_true", default=False, help="Save trained model (default: False).")
    parser.add_argument("--production_model", action="store_true", default=False, help="Production model (default: False).")
    parser.add_argument("--lora", action="store_true", default=False, help="Use LoRA (default: False).")
    parser.add_argument("--lora_r", type=int, default=8, help="Number of trainable parameters in the LoRA model.")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for the LoRA model.")
    parser.add_argument("--lora_dropout", type=float, default=0.01, help="Dropout rate for the LoRA model.")
    parser.add_argument("--sim_type", choices=["dot", "euclidean", "cosine"], default="dot", help="Cross-attention mechanism for token-parameter-attention")
    parser.add_argument("--token_attention", action="store_true", default=False, help="If true, use TokenFormer instead of Transformer blocks")

    # ----------------- ScikitArguments ----------------- #
    parser.add_argument("--scikit_n_iter", type=int, default=10, help="Number of iterations for scikit model.")
    parser.add_argument("--scikit_cv", type=int, default=3, help="Number of cross-validation folds for scikit model.")
    parser.add_argument("--scikit_random_state", type=int, default=None, help="Random state for scikit model (if None, uses global seed).")
    parser.add_argument("--scikit_model_name", type=str, default=None, help="Name of the scikit model to use.")
    parser.add_argument("--use_scikit", action="store_true", default=False, help="Use scikit model (default: False).")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processes to use in scikit.") # TODO integrate with GUI and main

    # ----------------- EmbeddingArguments ----------------- #
    parser.add_argument("--embedding_batch_size", type=int, default=16, help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", default=False, help="Whether to download embeddings (default: False).")
    parser.add_argument("--matrix_embed", action="store_true", default=False, help="Use matrix embedding (default: False).")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean", "var"], help="Pooling types for embeddings.")
    parser.add_argument("--save_embeddings", action="store_true", default=False, help="Save computed embeddings (default: False).")
    parser.add_argument("--embed_dtype", default="float32", help="Data type for embeddings.")
    parser.add_argument("--sql", action="store_true", default=False, help="Whether to use SQL storage (default: False).")
    parser.add_argument("--read_scaler", type=int, default=100, help="Read scaler for SQL storage.")
    
    # ----------------- Multi-Column Sequences ----------------- #
    parser.add_argument("--multi_column", nargs="+", default=None, help="If set, list of sequence column names to combine per sample.")

    # ----------------- TrainerArguments ----------------- #
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--probe_batch_size", type=int, default=64, help="Batch size for probe training.")
    parser.add_argument("--base_batch_size", type=int, default=4, help="Batch size for base model training.")
    parser.add_argument("--probe_grad_accum", type=int, default=1, help='Gradient accumulation steps for probe training.')
    parser.add_argument("--base_grad_accum", type=int, default=8, help='Gradient accumulation steps for base model training.')
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    ### TODO integrate
    #parser.add_argument("--probe_lr", type=float, default=1e-4, help="Learning rate for probe training.")
    #parser.add_argument("--base_lr", type=float, default=1e-5, help="Learning rate for base model training.")
    #parser.add_argument("--lr_scheduler", type=str, default='cosine', help='Learning rate scheduler.')
    #parser.add_argument("--optimizer", type=str, default='adamw', help='Optimizer.')
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=1, help="Patience for early stopping.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility (if omitted, current time is used).")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable deterministic behavior for reproducibility (will slow down training).")
    parser.add_argument("--full_finetuning", action="store_true", default=False, help="Full finetuning (default: False).")
    parser.add_argument("--hybrid_probe", action="store_true", default=False, help="Hybrid probe (default: False).")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of training runs with different seeds. Results will show mean±std across runs.")
    
    # ----------------- ProteinGym Arguments ----------------- #
    parser.add_argument("--dms_ids", nargs="+", default=["all"],
                        help="ProteinGym DMS assay IDs to evaluate (space-separated), or 'all' to run all assays.")
    parser.add_argument("--proteingym", action="store_true", default=False, help="ProteinGym (default: False).")
    parser.add_argument("--mode", type=str, default='benchmark',
                        help="ProteinGym zero-shot mode: 'benchmark', 'indels', 'multiples', 'singles'")
    parser.add_argument("--scoring_method", choices=["masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"], default="masked_marginal",
                        help="Select a scoring method for ProteinGym zero-shot.")
    parser.add_argument("--scoring_window", choices=["optimal", "sliding"], default="optimal",
                        help="Select how to slice the sequence for ProteinGym zero-shot.")
    parser.add_argument("--pg_batch_size", type=int, default=32,
                        help="Batch size for ProteinGym zero-shot scoring (default: 32).")
    parser.add_argument("--compare_scoring_methods", action="store_true", default=False,
                        help="Compare different scoring methods across models and DMS assays (default: False).")
    parser.add_argument("--score_only", action="store_true", default=False,
                        help="Only run the ProteinGym benchmarking script on existing CSV files, skip zero-shot scoring (default: False).")

    # ----------------- W&B Arguments ----------------- #
    parser.add_argument("--use_wandb_hyperopt", action="store_true", default=False, help="Use Weights & Biases hyperparameter optimization.")
    parser.add_argument("--wandb_project", type=str, default="Protify", help="W&B project name for sweeps.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user) for sweeps.")
    parser.add_argument("--sweep_config_path", type=str, default="yamls/sweep.yaml", help="Path to W&B sweep config YAML.")
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of hyperparameter trials to run in the sweep.")
    parser.add_argument("--sweep_method", type=str, default="bayes", choices=["bayes", "grid", "random"], help="Sweep method for hyperparameter optimization.")
    parser.add_argument("--sweep_metric_cls",type=str,default="eval_loss", help="Classification metric to optimize during sweep (e.g., eval_f1, eval_accuracy, eval_mcc)")
    parser.add_argument("--sweep_metric_reg",type=str,default="eval_loss", help="Regression metric to optimize during sweep (e.g., eval_r_squared, eval_spearman_rho, eval_pearson_rho)")
    parser.add_argument("--sweep_goal", type=str, default='minimize', choices=['maximize', 'minimize'], help="Goal for the sweep metric (maximize/minimize)")
    args = parser.parse_args()

    if args.hf_token is not None:
        from huggingface_hub import login
        # Override environment variable to ensure this token is used
        os.environ["HF_TOKEN"] = args.hf_token
        login(args.hf_token)
        print(f"Logged in to HuggingFace Hub with token from arguments")
    else:
        # Check if token exists in environment (from Modal secret or other source)
        try:
            hf_token_env = os.environ["HF_TOKEN"]
        except KeyError:
            hf_token_env = None
        if hf_token_env is not None and hf_token_env != "":
            print("Note: HF_TOKEN found in environment (from Modal secret or other source)")
            print("Note: This token will be used for read operations only unless overridden")
    if args.wandb_api_key is not None:
        try:
            import wandb
            wandb.login(key=args.wandb_api_key)
            print_message('Logged into Weights & Biases')
        except Exception as e:
            print_message(f'W&B login failed: {e}')
    if args.synthyra_api_key is not None:
        print_message('Synthyra API not integrated yet')

    if args.yaml_path is not None:
        with open(args.yaml_path, 'r') as file: 
            settings = yaml.safe_load(file)
        # Start from parser defaults, then overlay YAML values. This guarantees all expected attributes exist.
        base_args = parser.parse_args([])
        yaml_args = SimpleNamespace(**vars(base_args))
        for k, v in settings.items():
            setattr(yaml_args, k, v)
        yaml_args.hf_token = args.hf_token
        yaml_args.hf_home = args.hf_home
        yaml_args.synthyra_api_key = args.synthyra_api_key
        yaml_args.wandb_api_key = args.wandb_api_key
        yaml_args.use_wandb_hyperopt = args.use_wandb_hyperopt
        yaml_args.wandb_project = args.wandb_project
        yaml_args.wandb_entity = args.wandb_entity
        yaml_args.sweep_config_path = args.sweep_config_path
        yaml_args.sweep_count = args.sweep_count
        yaml_args.sweep_method = args.sweep_method
        yaml_args.sweep_metric_cls = args.sweep_metric_cls
        yaml_args.sweep_metric_reg = args.sweep_metric_reg
        yaml_args.sweep_goal = args.sweep_goal
        yaml_args.yaml_path = args.yaml_path
        return yaml_args
    else:
        return args


if __name__ == "__main__":
    # Settings that need to happen pre-imports
    args = parse_arguments()

    # Require that either datasets are specified or a ProteinGym experiment is chosen
    has_datasets = bool(args.data_names or args.data_dirs)
    has_proteingym = bool(args.proteingym)
    if not has_datasets and not has_proteingym:
        raise AssertionError("No datasets specified. Provide --data_names or --data_dirs, or run a ProteinGym experiment.")

    if args.hf_home is not None:
        # Needs to happen before any HF imports
        import pathlib
        base_path = args.hf_home
        cache_root = f"{base_path}/hf_cache"
        tmp_root   = f"{base_path}/tmp"
        pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"]            = cache_root
        os.environ["HF_DATASETS_CACHE"]  = f"{cache_root}/datasets"
        os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers" # this is deprecated, but does not hurt anything
        os.environ["HF_HUB_CACHE"]       = f"{cache_root}/hub"
        print(f"HF_HOME: {os.environ['HF_HOME']}")
        print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
        print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")

    # Set global seed before doing anything else
    # If seed is None, set_global_seed will derive it from current time
    if args.deterministic:
        from seed_utils import set_determinism
        set_determinism()


import torch
from torchinfo import summary

from probes.get_probe import ProbeArguments, get_probe
from base_models.get_base_models import BaseModelArguments, get_tokenizer, get_base_model_for_training
from base_models.utils import wrap_lora
from data.data_mixin import DataMixin, DataArguments
from probes.trainers import TrainerMixin, TrainerArguments
from probes.scikit_classes import ScikitArguments, ScikitProbe
from embedder import EmbeddingArguments, Embedder, get_embedding_filename
from logger import MetricsLogger, log_method_calls
from utils import torch_load, print_message, expand_dms_ids_all
from visualization.plot_result import create_plots
from hyperopt_utils import HyperoptModule
from benchmarks.proteingym.scorer import ProteinGymRunner
from benchmarks.proteingym.compare_scoring_methods import compare_scoring_methods
from seed_utils import set_global_seed


class MainProcess(MetricsLogger, DataMixin, TrainerMixin):
    def __init__(self, full_args, GUI=False):
        super(MainProcess, self).__init__(full_args)
        super(DataMixin, self).__init__()
        super(TrainerMixin, self).__init__()
        self.full_args = full_args
        if not GUI:
            self.start_log_main()

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            #"int8": torch.int8,
        }

    @log_method_calls
    def apply_current_settings(self):
        self.full_args.embed_dtype = self.dtype_map[self.full_args.embed_dtype]
        self.data_args = DataArguments(**self.full_args.__dict__)
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
        self.model_args = BaseModelArguments(**self.full_args.__dict__)
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        self.trainer_args = TrainerArguments(**self.full_args.__dict__)
        self.logger_args = SimpleNamespace(**self.full_args.__dict__)
        self.scikit_args = ScikitArguments(**self.full_args.__dict__)
        self._sql = self.full_args.sql
        self._full = self.full_args.matrix_embed
        self._max_length = self.full_args.max_length
        self._trim = self.full_args.trim
        self._delimiter = self.full_args.delimiter
        self._col_names = self.full_args.col_names
        self._multi_column = self.full_args.multi_column

    @log_method_calls
    def get_datasets(self):
        self.datasets, self.all_seqs = self.get_data()

    @log_method_calls
    def save_embeddings_to_disk(self):
        self.embedding_args.save_embeddings = True
        embedder = Embedder(self.embedding_args, self.all_seqs)
        for model_name in self.model_args.model_names:
            _ = embedder(model_name)

    def _create_model_factory(self, model_name, tokenwise, num_labels, hybrid):
        """Function for creating fresh models in multi-run mode."""
        def factory():
            model, _ = get_base_model_for_training(model_name, tokenwise=tokenwise, num_labels=num_labels, hybrid=hybrid)
            if self.probe_args.lora:
                model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
            return model
        return factory
    
    def _create_probe_factory(self):
        """Function for creating fresh probes in multi-run mode."""
        def factory():
            return get_probe(self.probe_args)
        return factory

    def _run_nn_probe(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            tokenizer,
            emb_dict=None,
            ppi=False,
            sweep_mode: bool = False,
        ):
        # Create initial probe (for single run or as template for multi-run)
        probe = get_probe(self.probe_args)
        summary(probe)
        
        # trainer_probe handles multi-run internally if num_runs > 1
        probe, valid_metrics, test_metrics, _, _ = self.trainer_probe(
            model=probe,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_dict,
            ppi=ppi,
            log_id=self.random_id,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return probe, valid_metrics, test_metrics

    def _train_nn_probe_fold(self, model_name, dms_id, subtrain_seqs, subtrain_labels,
                            valid_seqs, valid_labels, test_seqs, test_labels, 
                            emb_dict, fold_info):
        """Trains a neural network probe on a ProteinGym DMS assay CV fold."""

        train_set = {'seqs': subtrain_seqs, 'labels': subtrain_labels}
        valid_set = None if (valid_seqs is None or valid_labels is None) else {'seqs': valid_seqs, 'labels': valid_labels}
        test_set = {'seqs': test_seqs, 'labels': test_labels}
        
        # Get tokenizer and determine input dimensions
        tokenizer = get_tokenizer(model_name)
        
        if self._sql:
            save_path = os.path.join(self.embedding_args.embedding_save_dir, 
                                    f'{model_name}_{self._full}.db')
            input_dim = self.get_embedding_dim_sql(save_path, subtrain_seqs[0], tokenizer)
            emb_for_training = None
        else:
            save_path = os.path.join(self.embedding_args.embedding_save_dir,
                                    f'{model_name}_{self._full}.pth')
            emb_for_training = torch_load(save_path) if os.path.exists(save_path) else emb_dict
            input_dim = self.get_embedding_dim_pth(emb_for_training, subtrain_seqs[0], tokenizer)
        
        # Configure probe for regression
        self.probe_args.input_size = input_dim
        self.probe_args.task_type = 'regression'
        self.probe_args.num_labels = 1
        self.trainer_args.task_type = 'regression'
        
        probe = get_probe(self.probe_args)
        _, _, test_metrics = self.trainer_probe(
            model=probe,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=f"{dms_id}_{fold_info}",
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_for_training,
            ppi=False,
            log_id=f"{self.random_id}_{fold_info}",
        )
        
        # Handle both plain and test-prefixed metric keys returned by HF Trainer
        rho = None
        if 'spearman_rho' in test_metrics:
            rho = test_metrics['spearman_rho']
        elif 'test_spearman_rho' in test_metrics:
            rho = test_metrics['test_spearman_rho']

        mse = None
        if 'mse' in test_metrics:
            mse = test_metrics['mse']
        elif 'test_mse' in test_metrics:
            mse = test_metrics['test_mse']
        return rho, mse
    
    def _run_full_finetuning(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            ppi=False,
            sweep_mode: bool = False,
        ):
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        num_runs = self.trainer_args.num_runs
        
        model_factory = self._create_model_factory(model_name, tokenwise, num_labels, hybrid=False) if num_runs > 1 else None
        model, tokenizer = get_base_model_for_training(model_name, tokenwise=tokenwise, num_labels=num_labels, hybrid=False)
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        summary(model)
        model, valid_metrics, test_metrics, _, _ = self.trainer_base_model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            ppi=ppi,
            log_id=self.random_id,
            model_factory=model_factory,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model, valid_metrics, test_metrics

    def _run_hybrid_probe(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            tokenizer,
            emb_dict=None,
            ppi=False,
            sweep_mode: bool = False,
        ):
        # Random models don't have a trainable base model, so fall back to regular probe
        if "random" in model_name.lower():
            print_message(f"Model {model_name} does not support hybrid training. Training a linear probe instead.")
            probe = get_probe(self.probe_args)
            summary(probe)
            probe, valid_metrics, test_metrics = self.trainer_probe(
                model=probe,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_set,
                valid_dataset=valid_set,
                test_dataset=test_set,
                emb_dict=emb_dict,
                ppi=ppi,
                log_id=self.random_id,
            )
            if not sweep_mode:
                self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
                self.log_metrics(data_name, model_name, test_metrics, split_name='test')
            return probe, valid_metrics, test_metrics
        
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        num_runs = self.trainer_args.num_runs
        
        model_factory = self._create_model_factory(model_name, tokenwise, num_labels, hybrid=True) if num_runs > 1 else None
        probe_factory = self._create_probe_factory() if num_runs > 1 else None
        model, tokenizer = get_base_model_for_training(model_name, tokenwise=tokenwise, num_labels=num_labels, hybrid=True)
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        probe = get_probe(self.probe_args)
        summary(model)
        summary(probe)
        model, valid_metrics, test_metrics, _, _ = self.trainer_hybrid_model(
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_dict,
            ppi=ppi,
            log_id=self.random_id,
            model_factory=model_factory,
            probe_factory=probe_factory,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model, valid_metrics, test_metrics


    @log_method_calls
    def run_full_finetuning(self):
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        for model_name in self.model_args.model_names:
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {model_name}')
                _ = self._run_full_finetuning(model_name, data_name, train_set, valid_set, test_set, ppi)
                torch.cuda.empty_cache()

    @log_method_calls
    def run_hybrid_probes(self):
        probe_args = self.probe_args
        test_seq = self.all_seqs[0]

        # Log the combinations we're going to process
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        
        # for each model, gather the settings and embeddings
        # assumes save_embeddings_to_disk has already been called
        for model_name in self.model_args.model_names:
            self.logger.info(f"Processing model: {model_name}")
    
            # get tokenizer
            tokenizer = get_tokenizer(model_name)

            # get embedding size
            pooling_types = self.embedding_args.pooling_types
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                filename = get_embedding_filename(model_name, self._full, pooling_types, 'db')
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                input_size = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                filename = get_embedding_filename(model_name, self._full, pooling_types, 'pth')
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                emb_dict = torch_load(save_path)
                input_size = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            # Adjust input dim for multi-column vector embeddings
            if (not self._full) and self.full_args.multi_column:
                input_size = input_size * len(self.full_args.multi_column)

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_size = input_size * 2
                else:
                    probe_args.input_size = input_size
            
                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                ### TODO we currently need both, settings should probably be consolidated
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {model_name}')
                ### TODO eventually add options for optimizers and schedulers
                ### TODO here is probably where we can differentiate between the different training schemes
                _ = self._run_hybrid_probe(
                    model_name=model_name,
                    data_name=data_name,
                    train_set=train_set,
                    valid_set=valid_set,
                    test_set=test_set,
                    tokenizer=tokenizer,
                    emb_dict=emb_dict,
                    ppi=ppi,
                )
                torch.cuda.empty_cache()
                ### TODO may link from probe here to running inference on input csv or HF datasets

    @log_method_calls
    def run_nn_probes(self):
        probe_args = self.probe_args
        test_seq = self.all_seqs[0]

        # Log the combinations we're going to process
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        
        # for each model, gather the settings and embeddings
        # assumes save_embeddings_to_disk has already been called
        for model_name in self.model_args.model_names:
            self.logger.info(f"Processing model: {model_name}")
    
            # get tokenizer
            tokenizer = get_tokenizer(model_name)

            if 'custom' in model_name.lower():
                clean_model_name = model_name.split('---')[-1].split('/')[-1]
            else:
                clean_model_name = model_name

            # get embedding size
            pooling_types = self.embedding_args.pooling_types
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                filename = get_embedding_filename(clean_model_name, self._full, pooling_types, 'db')
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                input_size = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                filename = get_embedding_filename(clean_model_name, self._full, pooling_types, 'pth')
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                emb_dict = torch_load(save_path)
                input_size = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            # Adjust input dim for multi-column vector embeddings
            if (not self._full) and self.full_args.multi_column:
                input_size = input_size * len(self.full_args.multi_column)

            print(f'Input dim: {input_size}')

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_size = input_size * 2
                else:
                    probe_args.input_size = input_size
            
                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                ### TODO we currently need both, settings should probably be consolidated
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {clean_model_name}')
                ### TODO eventually add options for optimizers and schedulers
                ### TODO here is probably where we can differentiate between the different training schemes
                _ = self._run_nn_probe(
                    model_name=clean_model_name,
                    data_name=data_name,
                    train_set=train_set,
                    valid_set=valid_set,
                    test_set=test_set,
                    tokenizer=tokenizer,
                    emb_dict=emb_dict,
                    ppi=ppi,
                )
                torch.cuda.empty_cache()
                ### TODO may link from probe here to running inference on input csv or HF datasets

    @log_method_calls
    def run_scikit_scheme(self):    
        scikit_probe = ScikitProbe(self.scikit_args)
        for model_name in self.model_args.model_names:
            for data_name, dataset in self.datasets.items():
                ### find best scikit model and parameters via cross validation and lazy predict
                X_train, y_train, X_valid, y_valid, X_test, y_test, label_type = self.prepare_scikit_dataset(model_name, dataset)
                if label_type == 'singlelabel':
                    results = scikit_probe.find_best_classifier(X_train, y_train, X_valid, y_valid)
                elif label_type == 'regression':
                    results = scikit_probe.find_best_regressor(X_train, y_train, X_valid, y_valid)
                else:
                    raise ValueError(f'Label type {label_type} not supported')
                ### train and evaluate best model
                results = scikit_probe.run_specific_model(X_train, y_train, X_valid, y_valid, X_test, y_test, results)
    
    @log_method_calls
    def generate_plots(self):
        print_message("Generating visualization plots...")
        # Determine which results file to use
        results_file = os.path.join(self.full_args.results_dir, f"{self.random_id}.tsv")
        
        # Check if the results file exists
        if not os.path.exists(results_file):
            print_message(f"Results file not found: {results_file}")
            return
        
        # Get output directory
        output_dir = self.full_args.plots_dir

        print_message(f"Generating plots in {output_dir}...")
        create_plots(results_file, output_dir)
        print_message("Plots generated successfully!")
        
    def run_proteingym_zero_shot(self):
        """Run ProteinGym zero-shot for all specified models and DMS ids."""
        dms_ids = self.full_args.dms_ids
        mode = self.full_args.mode
        dms_ids = expand_dms_ids_all(dms_ids, mode=mode)
        if len(dms_ids) == 0:
            raise ValueError("--dms_ids is required when --proteingym is specified")
        model_names = self.full_args.model_names
        if len(model_names) == 0:
            raise ValueError("--model_names must specify at least one model")
        # Where to write results
        results_root = self.full_args.results_dir
        results_dir = os.path.join(results_root, 'proteingym')
        scoring_method = self.full_args.scoring_method
        scoring_window = self.full_args.scoring_window
        if isinstance(mode, str) and mode.lower() == 'indels':
            print_message("Only pll is currently supported for indels scoring.")
            scoring_method = 'pll'
        
        # Log the run
        self.logger.info(f"Running ProteinGym zero-shot with [{scoring_method}] scoring on {len(dms_ids)} DMS ids with models: {model_names}")
        
        runner = ProteinGymRunner(
            results_dir=results_dir,
            repo_id="GleghornLab/ProteinGym_DMS",
        )
        self._proteingym_timing = runner.run(
            dms_ids=dms_ids,
            model_names=model_names,
            mode=mode,
            scoring_method=scoring_method,
            scoring_window=scoring_window,
            batch_size=self.full_args.pg_batch_size,
        )
        print_message(f"ProteinGym zero-shot complete. Results in {results_dir}")

        # After all models are scored, run standardized performance benchmarking
        runner.run_benchmark(model_names, dms_ids, mode, scoring_method)

def main(args: SimpleNamespace):
    chosen_seed = set_global_seed(args.seed)
    args.seed = chosen_seed

    if args.replay_path is not None:
        from logger import LogReplayer
        replayer = LogReplayer(args.replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = args.replay_path
        # Overlay current CLI args (defaults) and then replay values, so all expected attributes exist.
        base_args = SimpleNamespace(**vars(args))
        merged = SimpleNamespace(**vars(base_args))
        for k, v in replay_args.__dict__.items():
            setattr(merged, k, v)
        replay_args = merged

        chosen_seed = set_global_seed(replay_args.seed, deterministic=replay_args.deterministic)
        replay_args.seed = chosen_seed
        main = MainProcess(replay_args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        replayer.run_replay(main)
    
    else:
        main = MainProcess(args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")

        if args.compare_scoring_methods and args.proteingym:
            # Run scoring method comparison
            print_message("Running scoring method comparison...")
            dms_ids = args.dms_ids
            mode = args.mode
            dms_ids = expand_dms_ids_all(dms_ids, mode=mode)
            model_names = args.model_names
            if len(model_names) == 0:
                raise ValueError("--model_names must specify at least one model")
            
            # Set up output path
            results_root = args.results_dir
            output_csv = os.path.join(results_root, 'scoring_methods_comparison.csv')
            
            summary_df = compare_scoring_methods(
                model_names=model_names,
                device=None,
                methods=None,
                dms_ids=dms_ids,
                progress=True,
                output_csv=output_csv
            )
            print_message(f"Scoring method comparison complete. Results saved to {output_csv}")
            return

        # Determine if current experiment passed datasets
        has_datasets = bool(args.data_names or args.data_dirs)

        # Run through datasets first (if any)
        if has_datasets:
          main.apply_current_settings()
          main.get_datasets()
          print_message(f"Number of sequences: {len(main.all_seqs)}")
          if main.full_args.use_wandb_hyperopt:
              if not main.full_args.full_finetuning:
                  main.save_embeddings_to_disk()
              HyperoptModule.run_wandb_hyperopt(main)

          elif main.full_args.full_finetuning:
              main.run_full_finetuning()

          elif main.full_args.hybrid_probe:
              main.save_embeddings_to_disk()
              main.run_hybrid_probes()

          elif main.full_args.use_scikit:
              main.save_embeddings_to_disk()
              main.run_scikit_scheme()
          else:
              main.save_embeddings_to_disk()
              main.run_nn_probes()
        else:
            # Determine if current experiment passed datasets
            has_datasets = bool(args.data_names or args.data_dirs)

            # Run through datasets first (if any)
            if has_datasets:
                main.apply_current_settings()
                main.get_datasets()
                num_seqs = len(main.all_seqs)
                print_message(f"Number of sequences: {num_seqs}")

                if main.full_args.full_finetuning:
                    main.run_full_finetuning()

                elif main.full_args.hybrid_probe:
                    main.save_embeddings_to_disk()
                    main.run_hybrid_probes()

                elif main.full_args.use_scikit:
                    main.save_embeddings_to_disk()
                    main.run_scikit_scheme()
                
                else:
                    main.save_embeddings_to_disk()
                    main.run_nn_probes()
            else:
                print_message("No datasets specified; proceeding with ProteinGym.")

            if args.proteingym:
                main.run_proteingym_zero_shot()
                try:
                    results_root = args.results_dir
                    results_dir = os.path.join(results_root, 'proteingym')
                    pg_scores = ProteinGymRunner.collect_spearman(results_dir, args.model_names)
                    for model_name, score in pg_scores.items():
                        if isinstance(score, (int, float)):
                            training_time = main._proteingym_timing[model_name]
                            metrics_dict = {'spearman': float(score)}
                            metrics_dict['training_time_seconds'] = float(training_time) if training_time is not None else None
                            main.log_metrics('proteingym', model_name, metrics_dict)
                except Exception as e:
                    print_message(f"Failed to log ProteinGym metrics: {e}")

        # Write results and generate plots
        main.write_results()
        main.generate_plots()
        main.end_log()

if __name__ == "__main__":
    main(args)
