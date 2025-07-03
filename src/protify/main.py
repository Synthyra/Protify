import os
import argparse
import yaml
from types import SimpleNamespace


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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
    parser.add_argument("--download_dir", default="Synthyra/mean_pooled_embeddings", help="Directory to download embeddings to.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots.")
    parser.add_argument("--replay_path", type=str, default=None, help="Path to the replay file.")
    parser.add_argument("--pretrained_probe_path", type=str, default=None) # TODO not used right now

    # ----------------- DataArguments ----------------- #
    parser.add_argument("--delimiter", default=",", help="Delimiter for data.")
    parser.add_argument("--col_names", nargs="+", default=["seqs", "labels"], help="Column names.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--trim", action="store_true", default=False,
                        help="Whether to trim sequences (default: False). If False, sequences are removed from the dataset if they are longer than max length. If True, they are truncated to max length."
                        )
    parser.add_argument("--data_names", nargs="+", default=["DeepLoc-2"], help="List of HF dataset names.") # TODO rename to data_names
    parser.add_argument("--data_dirs", nargs="+", default=[], help="List of local data directories.")

    # ----------------- BaseModelArguments ----------------- #
    parser.add_argument("--model_names", nargs="+", default=["ESM2-8"], help="List of model names to use.")

    # ----------------- ProbeArguments ----------------- #
    parser.add_argument("--probe_type", choices=["linear", "transformer", "retrievalnet", "lyra"], default="linear", help="Type of probe.")
    parser.add_argument("--tokenwise", action="store_true", default=False, help="Tokenwise probe (default: False).")
    ### TODO refactor to hidden_size
    parser.add_argument("--hidden_size", type=int, default=8192, help="Hidden dimension size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--pre_ln", action="store_false", default=True,
                        help="Disable pre-layernorm (default: enabled). Use --pre_ln to toggle off.")
    parser.add_argument("--classifier_dim", type=int, default=4096, help="Feed-forward dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for the transformer layers.")
    parser.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout rate for the classifier.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads in multi-head attention.")
    parser.add_argument("--rotary", action="store_false", default=True,
                        help="Disable rotary embeddings (default: enabled). Use --rotary to toggle off.")
    parser.add_argument("--probe_pooling_types", nargs="+", default=["cls", "mean"], help="Pooling types to use.")
    parser.add_argument("--save_model", action="store_true", default=False, help="Save trained model (default: False).")
    parser.add_argument("--production_model", action="store_true", default=False, help="Production model (default: False).")
    parser.add_argument("--lora", action="store_true", default=False, help="Use LoRA (default: False).")
    parser.add_argument("--lora_r", type=int, default=8, help="Number of trainable parameters in the LoRA model.")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for the LoRA model.")
    parser.add_argument("--lora_dropout", type=float, default=0.01, help="Dropout rate for the LoRA model.")


    parser.add_argument("--sim_type", choices=["dot", "euclidean", "cosine"], default="dot", help="Cross-attention mechanism for token-parameter-attention")
    parser.add_argument("--token_attention", action="store_true", default=False, help="If true, use TokenFormer instead of Transformer blocks")

    # ----------------- ScikitArguments ----------------- # # TODO add to GUI
    parser.add_argument("--scikit_n_iter", type=int, default=10, help="Number of iterations for scikit model.")
    parser.add_argument("--scikit_cv", type=int, default=3, help="Number of cross-validation folds for scikit model.")
    parser.add_argument("--scikit_random_state", type=int, default=42, help="Random state for scikit model.")
    parser.add_argument("--scikit_model_name", type=str, default=None, help="Name of the scikit model to use.")
    parser.add_argument("--use_scikit", action="store_true", default=False, help="Use scikit model (default: False).")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processes to use in scikit.") # TODO integrate with GUI and main

    # ----------------- EmbeddingArguments ----------------- #
    parser.add_argument("--embedding_batch_size", type=int, default=4, help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", default=False, help="Whether to download embeddings (default: False).")
    parser.add_argument("--matrix_embed", action="store_true", default=False, help="Use matrix embedding (default: False).")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean"], help="Pooling types for embeddings.")
    parser.add_argument("--save_embeddings", action="store_true", default=False, help="Save computed embeddings (default: False).")
    parser.add_argument("--embed_dtype", default="float32", help="Data type for embeddings.")
    parser.add_argument("--sql", action="store_true", default=False, help="Whether to use SQL storage (default: False).")
    parser.add_argument("--read_scaler", type=int, default=100, help="Read scaler for SQL storage.")

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
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument("--full_finetuning", action="store_true", default=False, help="Full finetuning (default: False).")
    parser.add_argument("--hybrid_probe", action="store_true", default=False, help="Hybrid probe (default: False).")

    args = parser.parse_args()

    if args.hf_token is not None:
        from huggingface_hub import login
        login(args.hf_token)
    if args.wandb_api_key is not None:
        print_message('Wandb not integrated yet')
    if args.synthyra_api_key is not None:
        print_message('Synthyra API not integrated yet')

    if args.yaml_path is not None:
        with open(args.yaml_path, 'r') as file: 
            settings = yaml.safe_load(file)
        yaml_args = SimpleNamespace(**settings)
        yaml_args.hf_token = args.hf_token
        yaml_args.hf_username = args.hf_username
        yaml_args.hf_home = args.hf_home
        yaml_args.synthyra_api_key = args.synthyra_api_key
        yaml_args.wandb_api_key = args.wandb_api_key
        yaml_args.yaml_path = args.yaml_path
        yaml_args.log_dir = args.log_dir
        yaml_args.results_dir = args.results_dir
        yaml_args.model_save_dir = args.model_save_dir
        return yaml_args
    else:
        return args

if __name__ == "__main__":
    args = parse_arguments()

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


import torch
from torchinfo import summary

from probes.get_probe import ProbeArguments, get_probe
from base_models.get_base_models import BaseModelArguments, get_tokenizer, get_base_model_for_training
from base_models.utils import wrap_lora
from data.data_mixin import DataMixin, DataArguments
from probes.trainers import TrainerMixin, TrainerArguments
from probes.scikit_classes import ScikitArguments, ScikitProbe
from embedder import EmbeddingArguments, Embedder
from logger import MetricsLogger, log_method_calls
from utils import torch_load, print_message
from visualization.plot_result import create_plots


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

    @log_method_calls
    def get_datasets(self):
        self.datasets, self.all_seqs = self.get_data()

    @log_method_calls
    def save_embeddings_to_disk(self):
        self.embedding_args.save_embeddings = True
        embedder = Embedder(self.embedding_args, self.all_seqs)
        for model_name in self.model_args.model_names:
            _ = embedder(model_name)

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
        ):
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
        self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
        self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return probe

    def _run_full_finetuning(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            ppi=False,
        ):
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        model, tokenizer = get_base_model_for_training(model_name, tokenwise=tokenwise, num_labels=num_labels, hybrid=False)
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        summary(model)
        model, valid_metrics, test_metrics = self.trainer_base_model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            ppi=ppi,
            log_id=self.random_id,
        )
        self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
        self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model

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
        ):
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        model, tokenizer = get_base_model_for_training(model_name, tokenwise=tokenwise, num_labels=num_labels, hybrid=True)
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        probe = get_probe(self.probe_args)
        summary(model)
        summary(probe)
        model, valid_metrics, test_metrics = self.trainer_hybrid_model(
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
        )
        self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
        self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model

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
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{self._full}.db')
                input_dim = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{self._full}.pth')
                emb_dict = torch_load(save_path)
                input_dim = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_dim = input_dim * 2
                else:
                    probe_args.input_dim = input_dim
            
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

            # get embedding size
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{self._full}.db')
                input_dim = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{self._full}.pth')
                emb_dict = torch_load(save_path)
                input_dim = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            print(f'Input dim: {input_dim}')

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_dim = input_dim * 2
                else:
                    probe_args.input_dim = input_dim
            
                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                ### TODO we currently need both, settings should probably be consolidated
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {model_name}')
                ### TODO eventually add options for optimizers and schedulers
                ### TODO here is probably where we can differentiate between the different training schemes
                _ = self._run_nn_probe(
                    model_name=model_name,
                    data_name=data_name,
                    train_set=train_set,
                    valid_set=valid_set,
                    test_set=test_set,
                    tokenizer=tokenizer,
                    emb_dict=emb_dict,
                    ppi=ppi,
                )
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
        

def main(args: SimpleNamespace):
    if args.replay_path is not None:
        from logger import LogReplayer
        replayer = LogReplayer(args.replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = args.replay_path
        main = MainProcess(replay_args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        replayer.run_replay(main)
    
    else:
        main = MainProcess(args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        main.apply_current_settings()
        main.get_datasets()
        print_message(f"Number of sequences: {len(main.all_seqs)}")
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
        main.write_results()
        main.generate_plots()
        main.end_log()


if __name__ == "__main__":
    main(args)